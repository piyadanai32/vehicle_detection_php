<?php
require_once __DIR__ . '/../controllers/VehicleController.php';

$uri = $_SERVER['REQUEST_URI'];
$method = $_SERVER['REQUEST_METHOD'];

$routes = [
    'GET' => [

    ],
    'POST' => [
        '#^/vehicle_count/?$#' => ['VehicleController', 'store'],
    ],
    'PUT' => [

    ],
    'DELETE' => [

    ]
];

$matched = false;
if (isset($routes[$method])) {
    foreach ($routes[$method] as $pattern => $handler) {
        if (preg_match($pattern, $uri)) {
            call_user_func($handler);
            $matched = true;
            break;
        }
    }
}

if (!$matched) {
    http_response_code(isset($routes[$method]) ? 404 : 405);
    echo json_encode(['error' => isset($routes[$method]) ? 'Not found' : 'Method not allowed']);
}
