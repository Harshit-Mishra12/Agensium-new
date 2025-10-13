-- Create customers table
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    signup_date TEXT
);

INSERT INTO customers (id, name, age, signup_date) VALUES
(1, 'Alice', 30, '2023-01-15'),
(2, 'Bob', 25, '2023-02-20'),
(3, 'Charlie', 35, '2023-03-05');

-- Create orders table
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount REAL,
    order_date TEXT
);

INSERT INTO orders (order_id, customer_id, amount, order_date) VALUES
(101, 1, 250.75, '2023-02-01'),
(102, 2, 100.00, '2023-02-15'),
(103, 1, 300.50, '2023-03-10');
