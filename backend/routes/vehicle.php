<?php
require_once __DIR__ . '/../controllers/VehicleController.php';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    VehicleController::store();
} else {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
}
