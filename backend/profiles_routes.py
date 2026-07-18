from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, List
import json
import io
from database import get_db
from auth import get_current_user

router = APIRouter(prefix="/api/user", tags=["profile"])

# ==================== Pydantic Models ====================

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    emergency_contact: Optional[str] = None
    medical_conditions_selected: Optional[List[str]] = None
    medical_conditions_notes: Optional[str] = None

class ProfileResponse(BaseModel):
    username: str
    email: Optional[str]
    full_name: Optional[str]
    phone: Optional[str]
    address: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    emergency_contact: Optional[str]
    medical_conditions: Optional[dict]
    profile_picture_url: Optional[str]
    profile_completed: bool

class MedicalCondition(BaseModel):
    id: int
    condition_name: str
    category: str

# ==================== Endpoints ====================

@router.get("/profile", response_model=ProfileResponse)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile"""
    try:
        user_id = current_user['user_id']
        
        with get_db() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT username, email, full_name, phone, address, age, gender,
                       emergency_contact, medical_conditions, profile_completed
                FROM users WHERE id = %s
            """, (user_id,))
            user = cur.fetchone()
            cur.close()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        medical_conditions = None
        if user['medical_conditions']:
            medical_conditions = json.loads(user['medical_conditions'])
        
        profile_picture_url = None
        if has_profile_picture(user_id):
            profile_picture_url = f"/api/user/profile/picture/{user_id}"
        
        return ProfileResponse(
            username=user['username'],
            email=user.get('email'),
            full_name=user.get('full_name'),
            phone=user.get('phone'),
            address=user.get('address'),
            age=user.get('age'),
            gender=user.get('gender'),
            emergency_contact=user.get('emergency_contact'),
            medical_conditions=medical_conditions,
            profile_picture_url=profile_picture_url,
            profile_completed=user.get('profile_completed', False)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/profile", response_model=dict)
async def update_profile(
    profile_data: ProfileUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update user profile"""
    try:
        user_id = current_user['user_id']
        
        # Build dynamic update query
        update_fields = []
        params = []
        
        if profile_data.full_name:
            update_fields.append("full_name = %s")
        params.append(profile_data.full_name)
        if profile_data.email:
            update_fields.append("email = %s")
            params.append(profile_data.email)
        if profile_data.phone:
            update_fields.append("phone = %s")
            params.append(profile_data.phone)
        if profile_data.address:
            update_fields.append("address = %s")
            params.append(profile_data.address)
        if profile_data.age:
            update_fields.append("age = %s")
            params.append(profile_data.age)
        if profile_data.gender:
            update_fields.append("gender = %s")
            params.append(profile_data.gender)
        if profile_data.emergency_contact:
            update_fields.append("emergency_contact = %s")
            params.append(profile_data.emergency_contact)
        
        # Handle medical conditions
        if profile_data.medical_conditions_selected or profile_data.medical_conditions_notes:
            medical_data = {
                "selected": profile_data.medical_conditions_selected or [],
                "notes": profile_data.medical_conditions_notes or ""
            }
            update_fields.append("medical_conditions = %s")
            params.append(json.dumps(medical_data))
        
        # Mark profile as completed if all fields filled
        if profile_data.full_name and profile_data.phone and profile_data.address:
            update_fields.append("profile_completed = TRUE")
        
        if not update_fields:
            return {"message": "No fields to update"}
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(user_id)
        
        query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = %s"
        
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            conn.commit()
            cur.close()
        
        return {"message": "Profile updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/profile/picture")
async def upload_profile_picture(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload profile picture (max 1MB)"""
    try:
        user_id = current_user['user_id']
        
        # Validate file size (1MB = 1048576 bytes)
        contents = await file.read()
        if len(contents) > 1048576:
            raise HTTPException(status_code=400, detail="File size exceeds 1MB limit")
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Only image files allowed (JPEG, PNG, GIF, WebP)")
        
        # Store in database
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE users SET profile_picture = %s WHERE id = %s", (contents, user_id))
            conn.commit()
            cur.close()
        
        return {"message": "Profile picture uploaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/picture/{user_id}")
async def get_profile_picture(user_id: int):
    """Get user profile picture"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT profile_picture FROM users WHERE id = %s", (user_id,))
            result = cur.fetchone()
            cur.close()
        
        if not result or not result[0]:
            raise HTTPException(status_code=404, detail="Profile picture not found")
        
        return StreamingResponse(
            io.BytesIO(result[0]),
            media_type="image/jpeg"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/profile/picture")
async def delete_profile_picture(current_user: dict = Depends(get_current_user)):
    """Delete profile picture"""
    try:
        user_id = current_user['user_id']
        
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE users SET profile_picture = NULL WHERE id = %s", (user_id,))
            conn.commit()
            cur.close()
        
        return {"message": "Profile picture deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/medical-conditions", response_model=List[MedicalCondition])
async def get_medical_conditions():
    """Get list of predefined medical conditions"""
    try:
        with get_db() as conn:
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT id, condition_name, category 
                FROM medical_conditions_reference 
                ORDER BY category, condition_name
            """)
            conditions = cur.fetchall()
            cur.close()
        
        return [
            MedicalCondition(id=c['id'], condition_name=c['condition_name'], category=c['category'])
            for c in conditions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Helper Functions ====================

def has_profile_picture(user_id: int) -> bool:
    """Check if user has profile picture"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT profile_picture FROM users WHERE id = %s", (user_id,))
            result = cur.fetchone()
            cur.close()
        return result and result[0] is not None
    except Exception:
        return False