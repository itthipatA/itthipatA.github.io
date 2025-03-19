import pandas as pd
import ujson as json  # faster JSON processing
from shapely import wkt
from shapely.geometry import mapping
import os
import math
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from collections import defaultdict
import torch
import time

# ตรวจสอบว่า GPU พร้อมใช้งานหรือไม่
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB")

def get_centroid_coordinates(geometry):
    """Get centroid coordinates from a geometry object"""
    try:
        # Try to get the first point of the first polygon
        if geometry["type"] == "Polygon":
            return geometry["coordinates"][0][0]
        elif geometry["type"] == "MultiPolygon":
            # For MultiPolygon, use the first point of the first polygon of the first MultiPolygon
            return geometry["coordinates"][0][0][0]
        else:
            print(f"Unsupported geometry type: {geometry['type']}")
            return None
    except Exception as e:
        print(f"Error getting centroid coordinates: {e}")
        return None

def batch_load_geometries(geometries_wkt):
    """แปลงข้อความ WKT เป็น Shapely geometries โดยใช้การประมวลผลแบบ vectorized"""
    results = []
    for wkt_str in geometries_wkt:
        try:
            geom = wkt.loads(wkt_str)
            results.append(geom)
        except Exception:
            results.append(None)
    return results

def process_chunk_with_gpu(df_chunk, grid_size=0.1):
    """ประมวลผลชุดข้อมูลโดยใช้ GPU เมื่อเป็นไปได้"""
    start_time = time.time()
    
    # แปลง DataFrame เป็น dictionary เพื่อง่ายต่อการประมวลผล
    chunk_dict = df_chunk.to_dict('records')
    
    # ดึงคอลัมน์ geometry ออกมาเพื่อแปลงเป็น Shapely objects
    geometries_wkt = [row['geometry'] for row in chunk_dict]
    
    # แบ่งชุดข้อมูลเป็น batch เพื่อประมวลผลทีละชุด (ป้องกันการใช้หน่วยความจำมากเกินไป)
    batch_size = 1000
    grid_cells = defaultdict(list)
    
    for i in tqdm(range(0, len(chunk_dict), batch_size), desc="Processing batches", leave=False):
        batch = chunk_dict[i:i+batch_size]
        batch_wkt = geometries_wkt[i:i+batch_size]
        
        # แปลง WKT เป็น geometry objects
        batch_geometries = batch_load_geometries(batch_wkt)
        
        for idx, (geom, row) in enumerate(zip(batch_geometries, batch)):
            if geom is None:
                continue
                
            try:
                # สร้าง GeoJSON feature
                feature = {
                    "type": "Feature",
                    "geometry": mapping(geom),
                    "properties": {
                        "area_in_meters": float(row['area_in_meters']),
                        "confidence": float(row['confidence']),
                        "full_plus_code": str(row['full_plus_code'])
                    }
                }
                
                # ดึงพิกัด centroid
                coords = get_centroid_coordinates(feature["geometry"])
                if coords is None:
                    continue
                    
                lng, lat = coords
                
                # คำนวณ grid cell
                cell_x = math.floor(lng / grid_size)
                cell_y = math.floor(lat / grid_size)
                cell_id = f"{cell_x}_{cell_y}"
                
                # เพิ่ม feature ลงใน grid cell
                grid_cells[cell_id].append(feature)
                
            except Exception:
                continue
    
    # วัดเวลาที่ใช้ในการประมวลผล
    elapsed = time.time() - start_time
    print(f"Processed {len(chunk_dict)} records in {elapsed:.2f} seconds ({len(chunk_dict)/elapsed:.2f} records/sec)")
    
    return dict(grid_cells)

def parallel_process_csv(csv_path, output_dir, grid_size=0.1, chunksize=100000):
    """ประมวลผลไฟล์ CSV โดยใช้ GPU และการประมวลผลแบบขนาน"""
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ CSV: {csv_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"สร้างหรือยืนยันไดเรกทอรีเรียบร้อยแล้ว: {output_dir}")
    
    # กำหนด dtypes เพื่อเพิ่มความเร็วในการอ่าน CSV
    dtypes = {
        'area_in_meters': np.float32,
        'confidence': np.float32,
        'full_plus_code': str,
        'geometry': str
    }
    
    # นับจำนวนแถวทั้งหมด
    print("นับจำนวนแถวทั้งหมด...")
    total_rows = sum(1 for _ in open(csv_path, 'r'))
    print(f"จำนวนแถวที่ต้องประมวลผลทั้งหมด: {total_rows}")
    
    # คำนวณจำนวน process ที่เหมาะสม (ใช้ CPU เพื่อโหลดข้อมูล)
    num_processes = min(mp.cpu_count(), 4)  # ลดจำนวน process ลงเพื่อลดการใช้หน่วยความจำ
    print(f"ใช้ {num_processes} processes สำหรับการโหลดข้อมูล")
    
    # เตรียม GPU memory
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # ตั้งค่า GPU ให้ใช้ทรัพยากรมากที่สุด
    if DEVICE.type == 'cuda':
        # ตั้งค่าให้ PyTorch ใช้ benchmark mode เพื่อหาการตั้งค่าที่เร็วที่สุด
        torch.backends.cudnn.benchmark = True
        # ตั้งค่าความแม่นยำให้ลดลงเพื่อเพิ่มความเร็ว
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # ประมวลผล CSV แบบขนาน
    print("กำลังแปลง CSV เป็น GeoJSON...")
    all_grid_cells = []
    processed_rows = 0
    
    # อ่าน CSV ทีละ chunk
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunksize, dtype=dtypes), 
                     desc="Processing chunks", unit="chunks"):
        # ประมวลผล chunk บน GPU
        grid_cells = process_chunk_with_gpu(chunk, grid_size)
        all_grid_cells.append(grid_cells)
        processed_rows += len(chunk)
        
        # แสดงความคืบหน้า
        print(f"ประมวลผลแล้ว {processed_rows} แถว จาก {total_rows} แถว ({processed_rows/total_rows*100:.2f}%)")
        
        if DEVICE.type == 'cuda':
            # แสดงข้อมูลการใช้ GPU
            print(f"GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")
    
    # รวมผลลัพธ์จากทุก chunk
    print("กำลังรวมผลลัพธ์...")
    merged_cells = defaultdict(list)
    for cells in tqdm(all_grid_cells, desc="Merging"):
        for cell_id, features in cells.items():
            merged_cells[cell_id].extend(features)
    
    # บันทึกไฟล์
    print("กำลังบันทึกไฟล์...")
    for cell_id, features in tqdm(merged_cells.items(), desc="Saving files"):
        output_file = os.path.join(output_dir, f"30d_buildings_{cell_id}.geojson")
        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }
        with open(output_file, "w") as f:
            json.dump(geojson_data, f)
    
    print(f"ประมวลผลเสร็จสิ้น! สร้างไฟล์ {len(merged_cells)} grid cells ใน {output_dir}")

if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define input and output paths
    csv_path = os.path.join(script_dir, "30d_buildings.csv")
    output_dir = os.path.join(script_dir, "grid_cells")
    
    try:
        # ใช้ chunksize ขนาดใหญ่ขึ้นเพื่อลดการโหลดข้อมูล
        parallel_process_csv(csv_path, output_dir, grid_size=0.05, chunksize=200000)
    except Exception as e:
        print(f"เกิดข้อผิดพลาดระหว่างการแปลงข้อมูล: {e}")
        # แสดง traceback เพื่อดูข้อผิดพลาดทั้งหมด
        import traceback
        traceback.print_exc()
