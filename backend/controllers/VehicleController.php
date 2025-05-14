<?php
require_once __DIR__ . '/../config/database.php';

class VehicleController {
    public static function store() {
        global $pdo;
        header('Content-Type: application/json');
        $data = json_decode(file_get_contents('php://input'), true);

        if (!isset($data['vehicle_type'], $data['direction'], $data['count'], $data['token'], $data['camera_id'])) {
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

            // Find camera_id (validate)
            $cameraStmt = $pdo->prepare("SELECT id FROM Camera WHERE id = ?");
            $cameraStmt->execute([$data['camera_id']]);
            $camera = $cameraStmt->fetch();
            if (!$camera) {
                throw new Exception("Invalid camera_id: " . $data['camera_id']);
            }

            // Insert into DetectionRecord (เพิ่ม camera_id)
            $stmt = $pdo->prepare("INSERT INTO DetectionRecord (edge_device_id, camera_id, vehicle_type_id, direction_type_id, count, time) VALUES (?, ?, ?, ?, ?, NOW())");
            $stmt->execute([
                $device['id'],
                $camera['id'],
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

    public static function summary() {
        global $pdo;
        header('Content-Type: application/json');
        $uri = $_SERVER['REQUEST_URI'];

        if (preg_match('#/vehicle_count/all/(camera|gate)=(\d+)&start=([\d\-]+)&stop=([\d\-]+)#', $uri, $matches)) {
            $type = $matches[1];
            $id = $matches[2];
            $start = $matches[3];
            $stop = $matches[4];

            try {
                $data = [];

                // ดึง vehicle_type_id และ direction_type_id ทั้งหมด
                $vehicleTypes = $pdo->query("SELECT id FROM VehicleType")->fetchAll(PDO::FETCH_COLUMN);
                $directionTypes = $pdo->query("SELECT id FROM DirectionType")->fetchAll(PDO::FETCH_COLUMN);

                if ($type === 'gate') {
                    // ดึงกล้องทั้งหมดใน gate นี้
                    $stmt = $pdo->prepare("SELECT id FROM Camera WHERE gate_id = ?");
                    $stmt->execute([$id]);
                    $camera_ids = $stmt->fetchAll(PDO::FETCH_COLUMN);

                    if (empty($camera_ids)) {
                        echo json_encode(['data' => []]);
                        return;
                    }

                    $in = str_repeat('?,', count($camera_ids) - 1) . '?';
                    $sql = "SELECT camera_id, vehicle_type_id, direction_type_id, SUM(count) as count
                            FROM DetectionRecord
                            WHERE camera_id IN ($in) AND time BETWEEN ? AND ?
                            GROUP BY camera_id, vehicle_type_id, direction_type_id";
                    $params = array_merge($camera_ids, [$start . " 00:00:00", $stop . " 23:59:59"]);
                    $stmt = $pdo->prepare($sql);
                    $stmt->execute($params);
                    $records = $stmt->fetchAll();

                    // group by camera_id
                    $grouped = [];
                    foreach ($camera_ids as $cid) {
                        $grouped[$cid] = [
                            'gate_id' => (int)$id,
                            'camera_id' => (int)$cid,
                            'start' => $start,
                            'stop' => $stop,
                            'details' => []
                        ];
                        // เตรียม details ทุก combination
                        foreach ($vehicleTypes as $vtid) {
                            foreach ($directionTypes as $dtid) {
                                $grouped[$cid]['details']["$vtid-$dtid"] = [
                                    'vehicle_type_id' => (int)$vtid,
                                    'direction_type_id' => (int)$dtid,
                                    'count' => 0
                                ];
                            }
                        }
                    }
                    // เติมค่าจาก records
                    foreach ($records as $row) {
                        $cid = $row['camera_id'];
                        $vtid = $row['vehicle_type_id'];
                        $dtid = $row['direction_type_id'];
                        $grouped[$cid]['details']["$vtid-$dtid"]['count'] = (int)$row['count'];
                    }
                    // push to data array
                    foreach ($grouped as $cam) {
                        // details เป็น array ไม่ใช่ associative
                        $cam['details'] = array_values($cam['details']);
                        $data[] = $cam;
                    }
                    echo json_encode(['data' => $data]);
                } else {
                    // กรณีเป็น camera โดยตรง
                    // ดึง gate_id ของกล้องนี้
                    $stmt = $pdo->prepare("SELECT gate_id FROM Camera WHERE id = ?");
                    $stmt->execute([$id]);
                    $gate_id = $stmt->fetchColumn();

                    $sql = "SELECT vehicle_type_id, direction_type_id, SUM(count) as count
                            FROM DetectionRecord
                            WHERE camera_id = ? AND time BETWEEN ? AND ?
                            GROUP BY vehicle_type_id, direction_type_id";
                    $params = [$id, $start . " 00:00:00", $stop . " 23:59:59"];
                    $stmt = $pdo->prepare($sql);
                    $stmt->execute($params);
                    $records = $stmt->fetchAll();

                    // เตรียม details ทุก combination
                    $details = [];
                    foreach ($vehicleTypes as $vtid) {
                        foreach ($directionTypes as $dtid) {
                            $details["$vtid-$dtid"] = [
                                'vehicle_type_id' => (int)$vtid,
                                'direction_type_id' => (int)$dtid,
                                'count' => 0
                            ];
                        }
                    }
                    // เติมค่าจาก records
                    foreach ($records as $row) {
                        $vtid = $row['vehicle_type_id'];
                        $dtid = $row['direction_type_id'];
                        $details["$vtid-$dtid"]['count'] = (int)$row['count'];
                    }
                    $data[] = [
                        'gate_id' => $gate_id !== false ? (int)$gate_id : null,
                        'camera_id' => (int)$id,
                        'start' => $start,
                        'stop' => $stop,
                        'details' => array_values($details)
                    ];
                    echo json_encode(['data' => $data]);
                }

            } catch (Exception $e) {
                http_response_code(500);
                echo json_encode(['error' => $e->getMessage()]);
            }

        } else {
            http_response_code(400);
            echo json_encode(['error' => 'Invalid parameters']);
            exit;
        }
    }

    public static function getAllDetectionRecord() {
        global $pdo;
        header('Content-Type: application/json');
        try {
            $sql = "SELECT * FROM DetectionRecord";
            $stmt = $pdo->query($sql);
            $records = $stmt->fetchAll(PDO::FETCH_ASSOC);
            echo json_encode(['data' => $records]);
        } catch (Exception $e) {
            http_response_code(500);
            echo json_encode(['error' => $e->getMessage()]);
        }
    }
}