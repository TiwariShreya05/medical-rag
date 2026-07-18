import React, { useState, useEffect } from 'react';
import './Profile.css';

const Profile = () => {
  const [profile, setProfile] = useState({
    username: '',
    email: '',
    full_name: '',
    phone: '',
    address: '',
    age: '',
    gender: '',
    emergency_contact: ''
  });

  const [profilePicture, setProfilePicture] = useState(null);
  const [profilePicturePreview, setProfilePicturePreview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [editMode, setEditMode] = useState(false);

  const token = localStorage.getItem('token');
  const BASE_URL = 'http://127.0.0.1:8000';

  // Load profile data
  useEffect(() => {
    const loadProfile = async () => {
      try {
        const profileRes = await fetch(`${BASE_URL}/api/user/profile`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        if (profileRes.ok) {
          const data = await profileRes.json();
          console.log('Loaded profile:', data);
          setProfile({
            username: data.username || '',
            email: data.email || '',
            full_name: data.full_name || '',
            phone: data.phone || '',
            address: data.address || '',
            age: data.age || '',
            gender: data.gender || '',
            emergency_contact: data.emergency_contact || ''
          });
          
          if (data.profile_picture_url) {
            setProfilePicturePreview(data.profile_picture_url);
          }
        }
      } catch (error) {
        console.error('Error loading profile:', error);
        setMessage({ type: 'error', text: 'Failed to load profile' });
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, [token]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setProfile(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleProfilePictureChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 1048576) {
        setMessage({ type: 'error', text: 'File size exceeds 1MB limit' });
        return;
      }
      setProfilePicture(file);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setProfilePicturePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSaveProfile = async () => {
    setSaving(true);
    setMessage({ type: '', text: '' });

    try {
      // Upload profile picture if changed
      if (profilePicture) {
        const formData = new FormData();
        formData.append('file', profilePicture);
        const picRes = await fetch(`${BASE_URL}/api/user/profile/picture`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${token}` },
          body: formData
        });
        if (!picRes.ok) throw new Error('Failed to upload picture');
      }

      // Update profile
      const updateData = {
        full_name: profile.full_name,
        email: profile.email,
        phone: profile.phone,
        address: profile.address,
        age: profile.age ? parseInt(profile.age) : null,
        gender: profile.gender,
        emergency_contact: profile.emergency_contact
      };

      console.log('Sending update:', updateData);

      const res = await fetch(`${BASE_URL}/api/user/profile`, {
        method: 'PUT',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(updateData)
      });

      if (res.ok) {
        setMessage({ type: 'success', text: 'Profile saved successfully!' });
        setEditMode(false);
        setProfilePicture(null);
        
        // Reload profile to confirm save
        setTimeout(() => {
          window.location.reload();
        }, 1500);
      } else {
        const error = await res.json();
        console.error('Save error:', error);
        setMessage({ type: 'error', text: error.detail || 'Failed to save profile' });
      }
    } catch (error) {
      console.error('Error saving profile:', error);
      setMessage({ type: 'error', text: error.message });
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return <div style={{ padding: '40px', color: '#666' }}>Loading profile...</div>;
  }

  return (
    <div className="profile-simple">
      {/* Header */}
      <div className="profile-top">
        <div>
          <h1>Profile</h1>
          <p>Update your personal information</p>
        </div>
        <button
          className={`btn-edit ${editMode ? 'cancel' : ''}`}
          onClick={() => setEditMode(!editMode)}
        >
          {editMode ? 'Cancel' : 'Edit'}
        </button>
      </div>

      {/* Alert */}
      {message.text && (
        <div className={`alert ${message.type}`}>
          {message.text}
        </div>
      )}

      {/* Content */}
      <div className="profile-main">
        {/* Photo */}
        <div className="photo-section">
          <div className="photo">
            {profilePicturePreview ? (
              <img src={profilePicturePreview} alt="Profile" />
            ) : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
            )}
          </div>
          {editMode && (
            <>
              <input
                type="file"
                id="pic-input"
                accept="image/*"
                onChange={handleProfilePictureChange}
                style={{ display: 'none' }}
              />
              <label htmlFor="pic-input" className="btn-photo">Upload Photo</label>
              <p className="photo-info">Max 1MB</p>
            </>
          )}
        </div>

        {/* Form */}
        <div className="form-section">
          <div className="form-row">
            <div className="form-field">
              <label>Username</label>
              <input type="text" value={profile.username} disabled readOnly />
            </div>
            <div className="form-field">
              <label>Email</label>
              <input
                type="email"
                name="email"
                value={profile.email}
                onChange={handleInputChange}
                disabled={!editMode}
                placeholder="your@email.com"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-field full-width">
              <label>Full Name</label>
              <input
                type="text"
                name="full_name"
                value={profile.full_name}
                onChange={handleInputChange}
                disabled={!editMode}
                placeholder="Your full name"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-field">
              <label>Phone</label>
              <input
                type="tel"
                name="phone"
                value={profile.phone}
                onChange={handleInputChange}
                disabled={!editMode}
                placeholder="9876543210"
              />
            </div>
            <div className="form-field">
              <label>Age</label>
              <input
                type="number"
                name="age"
                value={profile.age}
                onChange={handleInputChange}
                disabled={!editMode}
                placeholder="25"
                min="0"
                max="150"
              />
            </div>
            <div className="form-field">
              <label>Gender</label>
              <select
                name="gender"
                value={profile.gender}
                onChange={handleInputChange}
                disabled={!editMode}
              >
                <option value="">Select</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-field full-width">
              <label>Address</label>
              <textarea
                name="address"
                value={profile.address}
                onChange={handleInputChange}
                disabled={!editMode}
                placeholder="Your address"
                rows="3"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-field full-width">
              <label>Emergency Contact</label>
              <input
                type="text"
                name="emergency_contact"
                value={profile.emergency_contact}
                onChange={handleInputChange}
                disabled={!editMode}
                placeholder="Name - 9876543210"
              />
            </div>
          </div>

          {editMode && (
            <div className="form-row">
              <button className="btn-save" onClick={handleSaveProfile} disabled={saving}>
                {saving ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Profile;