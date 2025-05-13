<?php
require_once __DIR__ . '/../config/database.php';

class VehicleController {
    public static function store() {
        global $pdo;
        $data = json_decode(file_get_contents('php://input'), true);

        // Check for required fields including token
        if (!isset($data['vehicle_type'], $data['direction'], $data['count'], $data['token'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid input']);
            exit;
        }

        try {
            // Find vehicle_type_id
            $vehicleTypeStmt = $pdo->prepare("SELECT id FROM VehicleType WHERE name = ?");
            $vehicleTypeStmt->execute([$data['vehicle_type']]);
            $vehicleType = $vehicleTypeStmt->fetch();
            if (!$vehicleType) {
                throw new Exception("Invalid vehicle type: " . $data['vehicle_type']);
            }

            // Find direction_type_id
            $directionTypeStmt = $pdo->prepare("SELECT id FROM DirectionType WHERE name = ?");
            $directionTypeStmt->execute([$data['direction']]);
            $directionType = $directionTypeStmt->fetch();
            if (!$directionType) {
                throw new Exception("Invalid direction: " . $data['direction']);
            }

            // Find edge_device_id by token
            $deviceStmt = $pdo->prepare("SELECT id FROM EdgeDevice WHERE token = ?");
            $deviceStmt->execute([$data['token']]);
            $device = $deviceStmt->fetch();
            if (!$device) {
                throw new Exception("Invalid or missing device token");
            }

            // Insert into DetectionRecord
            $stmt = $pdo->prepare("INSERT INTO DetectionRecord (edge_device_id, vehicle_type_id, direction_type_id, count, time) VALUES (?, ?, ?, ?, NOW())");
            $stmt->execute([
                $device['id'],
                $vehicleType['id'],
                $directionType['id'],
                $data['count']
            ]);

            // Return edge_device_id in response
            echo json_encode(['status' => 'success', 'edge_device_id' => $device['id']]);

        } catch (Exception $e) {
            http_response_code(500);
            echo json_encode(['error' => $e->getMessage()]);
        }
    }
}