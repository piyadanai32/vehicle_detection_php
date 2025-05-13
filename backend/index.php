<?php

$request_uri = $_SERVER['REQUEST_URI'];

if (preg_match('#^/vehicle_count#', $request_uri)) {
    require __DIR__ . '/routes/vehicle.php';
    exit;
}