CREATE DATABASE IF NOT EXISTS cctv_traffic_db;
USE cctv_traffic_db;

-- สร้างตาราง EdgeDevice
CREATE TABLE IF NOT EXISTS EdgeDevice (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    token VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- สร้างตาราง VehicleType
CREATE TABLE IF NOT EXISTS VehicleType (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

-- สร้างตาราง DirectionType
CREATE TABLE IF NOT EXISTS DirectionType (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

-- สร้างตาราง DetectionRecord
CREATE TABLE IF NOT EXISTS DetectionRecord (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    edge_device_id BIGINT NOT NULL,
    time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vehicle_type_id BIGINT NOT NULL,
    direction_type_id BIGINT NOT NULL,
    count INT NOT NULL,
    FOREIGN KEY (edge_device_id) REFERENCES EdgeDevice(id),
    FOREIGN KEY (vehicle_type_id) REFERENCES VehicleType(id),
    FOREIGN KEY (direction_type_id) REFERENCES DirectionType(id)
);

-- เพิ่มข้อมูลเริ่มต้น
INSERT INTO VehicleType (name) VALUES ('car'), ('motorcycle'), ('bus');
INSERT INTO DirectionType (name) VALUES ('in'), ('out');
INSERT INTO EdgeDevice (name, location, token) VALUES ('CCTV-Edge-01', 'Main Entrance', 'dajsdkasjdsuad2348werwerewfjslfj8w424');