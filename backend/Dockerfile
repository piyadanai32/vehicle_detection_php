FROM php:8.1-cli

# ติดตั้ง PDO MySQL และ Composer
RUN docker-php-ext-install pdo pdo_mysql

# ติดตั้ง Composer
COPY --from=composer:2.6 /usr/bin/composer /usr/bin/composer

# คัดลอก source code
WORKDIR /app
COPY . /app

# ติดตั้ง dependencies PHP
RUN composer install

# เปิด port 8000
EXPOSE 8000

# คำสั่งรัน PHP built-in server
CMD ["php", "-S", "0.0.0.0:8000", "index.php"]
