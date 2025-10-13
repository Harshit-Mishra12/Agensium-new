-- Baseline database dump

CREATE TABLE customers
(
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    country TEXT,
    signup_date DATE
);

INSERT INTO customers
    (id, name, age, country, signup_date)
VALUES
    (1, 'Alice', 30, 'USA', '2020-01-15'),
    (2, 'Bob', 25, 'Canada', '2020-03-22'),
    (3, 'Charlie', 35, 'USA', '2020-05-10'),
    (4, 'Diana', 28, 'UK', '2020-07-01');

CREATE TABLE orders
(
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount REAL,
    status TEXT,
    order_date DATE
);

INSERT INTO orders
    (order_id, customer_id, amount, status, order_date)
VALUES
    (101, 1, 250.0, 'completed', '2020-02-01'),
    (102, 2, 150.0, 'completed', '2020-04-12'),
    (103, 3, 400.0, 'pending', '2020-06-15');
