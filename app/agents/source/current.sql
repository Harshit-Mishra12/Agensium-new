-- Current database dump with drift

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
    (1, 'Alice', 31, 'USA', '2020-01-15'),
    -- Age drift
    (2, 'Bob', 25, 'Canada', '2020-03-22'),
    (3, 'Charlie', 36, 'Mexico', '2020-05-10'),
    -- Country drift
    (4, 'Diana', 28, 'UK', '2020-07-01'),
    (5, 'Eva', 29, 'USA', '2020-08-20');
-- New record

CREATE TABLE orders
(
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount REAL,
    status TEXT,
    order_date DATE,
    payment_method TEXT
    -- New column drift
);

INSERT INTO orders
    (order_id, customer_id, amount, status, order_date, payment_method)
VALUES
    (101, 1, 260.0, 'completed', '2020-02-01', 'credit_card'),
    -- Amount drift
    (102, 2, 150.0, 'completed', '2020-04-12', 'paypal'),
    (103, 3, 400.0, 'shipped', '2020-06-15', 'credit_card'),
    -- Status drift
    (104, 5, 120.0, 'completed', '2020-09-05', 'debit_card');  -- New row
