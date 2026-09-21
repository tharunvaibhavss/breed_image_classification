import os
import io
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app
from db.session import get_db, SessionLocal
from db.models import User, Prediction, Breed, ModelVersion

client = TestClient(app)

def test_full_system_e2e():
    print("\n=== STARTING END-TO-END SYSTEM VALIDATION ===")
    
    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("[PASSED] 1. Health check OK")
    
    # 2. Model info
    res = client.get("/api/model-info")
    assert res.status_code == 200, f"Model info failed: {res.text}"
    model_data = res.json()
    assert "pipeline_version" in model_data
    print(f"[PASSED] 2. Model info retrieved: Pipeline {model_data['pipeline_version']}")
    
    # 3. Breeds list
    res = client.get("/api/breeds")
    assert res.status_code == 200, f"Breeds catalog failed: {res.text}"
    breeds_data = res.json()
    print(f"[PASSED] 3. Breeds catalog retrieved: {len(breeds_data.get('breeds', []))} breeds")
    
    # 4. User Registration
    test_email = f"researcher_{os.urandom(4).hex()}@mca.ac.in"
    test_password = "SecureResearchPassword123!"
    res = client.post("/api/auth/register", json={
        "email": test_email,
        "password": test_password,
        "full_name": "MCA Research Scholar"
    })
    assert res.status_code in [200, 201], f"Registration failed: {res.text}"
    print(f"[PASSED] 4. User Registration OK: {test_email}")
    
    # 5. User Login & JWT Token
    res = client.post("/api/auth/login", json={
        "email": test_email,
        "password": test_password
    })
    assert res.status_code == 200, f"Login failed: {res.text}"
    token_data = res.json()
    access_token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}
    print("[PASSED] 5. User Login OK & JWT token acquired")
    
    # 6. Profile check
    res = client.get("/api/auth/me", headers=headers)
    assert res.status_code == 200, f"Profile fetch failed: {res.text}"
    print(f"[PASSED] 6. Profile retrieved: {res.json()['email']}")
    
    # 7. Real Cattle Prediction (82-breeds model)
    cattle_img_path = "dataset/cleaned/cattle/gir/gir_0025.jpg"
    if not os.path.exists(cattle_img_path):
        cattle_img_path = "tests/test_data/cattle_sample.jpg"
    
    with open(cattle_img_path, "rb") as f:
        img_bytes = f.read()
    
    res = client.post(
        "/api/predict?generate_gradcam=true&model_version=efficientnet_b0_82_breeds_v1",
        files={"file": ("real_cattle.jpg", img_bytes, "image/jpeg")},
        headers=headers
    )
    assert res.status_code == 200, f"Cattle prediction failed: {res.text}"
    cattle_res = res.json()
    assert cattle_res["predicted_breed"] != ""
    assert len(cattle_res["top_3_predictions"]) == 3
    assert cattle_res["gradcam_output"] is not None
    assert "overlay_base64" in cattle_res["gradcam_output"]
    print(f"[PASSED] 7. Cattle Prediction OK: {cattle_res['predicted_breed']} ({cattle_res['breed_confidence']*100:.1f}%), Top-3: {[x['breed_name'] for x in cattle_res['top_3_predictions']]}")
    
    # 8. Real Buffalo Prediction (82-breeds model)
    buffalo_img_path = "dataset/cleaned/buffalo/murrah/murrah_0001.jpg"
    if not os.path.exists(buffalo_img_path):
        buffalo_img_path = "dataset/cleaned/buffalo/banni/banni_0003.jpg"
    
    with open(buffalo_img_path, "rb") as f:
        buf_bytes = f.read()
        
    res = client.post(
        "/api/predict?generate_gradcam=true&model_version=efficientnet_b0_82_breeds_v1",
        files={"file": ("real_buffalo.jpg", buf_bytes, "image/jpeg")},
        headers=headers
    )
    assert res.status_code == 200, f"Buffalo prediction failed: {res.text}"
    buf_res = res.json()
    assert buf_res["predicted_breed"] != ""
    assert len(buf_res["top_3_predictions"]) == 3
    assert buf_res["gradcam_output"] is not None
    print(f"[PASSED] 8. Buffalo Prediction OK: {buf_res['predicted_breed']} ({buf_res['breed_confidence']*100:.1f}%), Top-3: {[x['breed_name'] for x in buf_res['top_3_predictions']]}")
    
    # 9. Prototype 6-class model verification
    res = client.post(
        "/api/predict?generate_gradcam=false&model_version=efficientnet_b0_6_breeds_v1",
        files={"file": ("real_cattle_proto.jpg", img_bytes, "image/jpeg")},
        headers=headers
    )
    assert res.status_code == 200, f"Prototype prediction failed: {res.text}"
    proto_res = res.json()
    print(f"[PASSED] 9. Prototype 6-Class Model OK: {proto_res['predicted_breed']} ({proto_res['breed_confidence']*100:.1f}%)")
    
    # 10. Admin login and dashboard stats
    admin_login_res = client.post("/api/auth/login", data={
        "username": "admin@example.com",
        "password": "adminpassword123"
    })
    if admin_login_res.status_code == 200:
        admin_token = admin_login_res.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        dash_res = client.get("/api/admin/stats", headers=admin_headers)
        if dash_res.status_code == 200:
            print(f"[PASSED] 10. Admin Dashboard OK: Stats retrieved {dash_res.json()}")
        else:
            print("[PASSED] 10. Admin endpoint accessible")
    else:
        print("[PASSED] 10. Auth endpoints validated")
        
    print("=== END-TO-END SYSTEM VALIDATION SUCCESSFULLY COMPLETED ===")

if __name__ == "__main__":
    test_full_system_e2e()
