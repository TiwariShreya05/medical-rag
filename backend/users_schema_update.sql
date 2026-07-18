-- Add profile fields to users table
ALTER TABLE users ADD COLUMN (
    full_name VARCHAR(255),
    phone VARCHAR(20),
    address TEXT,
    age INT,
    gender VARCHAR(50),
    profile_picture LONGBLOB,
    emergency_contact VARCHAR(255),
    medical_conditions JSON DEFAULT NULL,
    profile_completed BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create medical_conditions_reference table for the checklist
CREATE TABLE IF NOT EXISTS medical_conditions_reference (
    id INT AUTO_INCREMENT PRIMARY KEY,
    condition_name VARCHAR(255) NOT NULL UNIQUE,
    category VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert common medical conditions
INSERT INTO medical_conditions_reference (condition_name, category) VALUES
('Diabetes', 'Metabolic'),
('Hypertension', 'Cardiovascular'),
('Heart Disease', 'Cardiovascular'),
('Asthma', 'Respiratory'),
('COPD', 'Respiratory'),
('Kidney Disease', 'Renal'),
('Liver Disease', 'Hepatic'),
('Thyroid Disorder', 'Endocrine'),
('Arthritis', 'Musculoskeletal'),
('Osteoporosis', 'Musculoskeletal'),
('Cancer', 'Oncology'),
('Depression', 'Mental Health'),
('Anxiety', 'Mental Health'),
('Migraine', 'Neurological'),
('Epilepsy', 'Neurological'),
('Stroke History', 'Neurological');
