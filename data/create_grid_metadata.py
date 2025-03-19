import os
import json
import glob
import re
from tqdm import tqdm

def create_grid_metadata(grid_cells_dir, output_file, grid_size=0.1):
    """
    สร้างไฟล์ grid_metadata.json โดยการสแกนไฟล์ GeoJSON ทั้งหมดในโฟลเดอร์
    
    Args:
        grid_cells_dir (str): พาธของโฟลเดอร์ที่มีไฟล์ GeoJSON
        output_file (str): พาธของไฟล์ metadata ที่จะสร้าง
        grid_size (float): ขนาดของกริด (ตามที่ใช้ในการแบ่งข้อมูล)
    """
    print(f"เริ่มสร้างไฟล์ grid_metadata.json จากไฟล์ในโฟลเดอร์ {grid_cells_dir}")
    
    # ตรวจสอบว่าโฟลเดอร์มีอยู่จริง
    if not os.path.exists(grid_cells_dir):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {grid_cells_dir}")
    
    # ค้นหาไฟล์ GeoJSON ทั้งหมด
    geojson_files = glob.glob(os.path.join(grid_cells_dir, "*.geojson"))
    print(f"พบไฟล์ GeoJSON จำนวน {len(geojson_files)} ไฟล์")
    
    # สร้างโครงสร้างข้อมูลสำหรับ metadata
    metadata = {
        "totalFeatures": 0,
        "cells": {},
        "gridSize": grid_size,
        "version": "1.0",
        "dataSource": "Google Open Buildings - Lampang, Thailand"
    }
    
    # สแกนแต่ละไฟล์ GeoJSON
    for geojson_file in tqdm(geojson_files, desc="กำลังสแกนไฟล์"):
        # ดึง cell_id จากชื่อไฟล์
        file_name = os.path.basename(geojson_file)
        match = re.search(r'30d_buildings_(\d+)_(\d+)\.geojson', file_name)
        
        if not match:
            print(f"ข้ามไฟล์ {file_name} เนื่องจากไม่ตรงกับรูปแบบชื่อไฟล์ที่คาดหวัง")
            continue
        
        cell_x, cell_y = int(match.group(1)), int(match.group(2))
        cell_id = f"{cell_x}_{cell_y}"
        
        # คำนวณขอบเขตของเซลล์
        bounds = {
            "north": (cell_y + 1) * grid_size,
            "south": cell_y * grid_size,
            "east": (cell_x + 1) * grid_size,
            "west": cell_x * grid_size
        }
        
        try:
            # อ่านไฟล์ GeoJSON และนับจำนวน features
            with open(geojson_file, 'r') as f:
                geojson_data = json.load(f)
                feature_count = len(geojson_data.get("features", []))
            
            # เพิ่มข้อมูลลงใน metadata
            metadata["cells"][cell_id] = {
                "filename": file_name,
                "featureCount": feature_count,
                "bounds": bounds
            }
            
            # อัปเดตจำนวน features ทั้งหมด
            metadata["totalFeatures"] += feature_count
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file_name}: {e}")
    
    # บันทึกไฟล์ metadata
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"สร้างไฟล์ {output_file} เสร็จสมบูรณ์")
    print(f"พบ {len(metadata['cells'])} เซลล์ ที่มีอาคารทั้งหมด {metadata['totalFeatures']} อาคาร")

if __name__ == "__main__":
    # ตำแหน่งของสคริปต์
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # กำหนดพาธของโฟลเดอร์และไฟล์
    grid_cells_dir = os.path.join(script_dir, "grid_cells")
    output_file = os.path.join(grid_cells_dir, "grid_metadata.json")
    
    try:
        # สร้างไฟล์ metadata โดยใช้ขนาดกริดเดียวกับที่ใช้ในการแบ่งข้อมูล
        # (ปรับขนาดกริดตามที่ใช้ในสคริปต์ CSVtoGeoJson.py)
        create_grid_metadata(grid_cells_dir, output_file, grid_size=0.05)
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการสร้างไฟล์ metadata: {e}")
        import traceback
        traceback.print_exc() 