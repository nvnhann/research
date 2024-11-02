/*
Lists each movie, the film category it is classified in, and the number of times it has been rented out.
*/
SELECT f.title AS movie_title, 
       c.name AS category_name, 
       COUNT(r.rental_id) AS rental_count
FROM film f
JOIN film_category fc ON f.film_id = fc.film_id
JOIN category c ON fc.category_id = c.category_id
LEFT JOIN inventory i ON f.film_id = i.film_id
LEFT JOIN rental r ON i.inventory_id = r.inventory_id
WHERE c.name IN ('Animation', 'Children', 'Classics', 'Comedy', 'Family', 'Music')
GROUP BY f.film_id, c.name
HAVING COUNT(r.rental_id) > 0
ORDER BY c.name;

-- v2

SELECT f.title AS movie_title, 
       c.name AS category_name, 
       ( SELECT COUNT(*) 
            FROM rental r 
            JOIN inventory i ON r.inventory_id = i.inventory_id 
            WHERE i.film_id = f.film_id
        ) AS rental_count
FROM film f
JOIN film_category fc ON f.film_id = fc.film_id
JOIN category c ON fc.category_id = c.category_id
WHERE c.name IN ('Animation', 'Children', 'Classics', 'Comedy', 'Family', 'Music')
AND EXISTS (
    SELECT 1 FROM rental r
    JOIN inventory i ON r.inventory_id = i.inventory_id
    WHERE i.film_id = f.film_id
)
GROUP BY f.film_id, c.name
ORDER BY c.name;


-- Question 2
/*
The movie titles and divide them into 4 levels (first_quarter, second_quarter, third_quarter, and final_quarter) based on the quartiles (25%, 50%, 75%) of the average rental duration(in the number of days) for movies across all categories?
*/
WITH RentalDurationQuartiles AS (
    SELECT 
        f.title,
        fc.category_id,
        c.name,
        f.rental_duration,
        NTILE(4) OVER (ORDER BY f.rental_duration) AS quartile
    FROM 
        film f
    JOIN film_category fc ON f.film_id = fc.film_id
    JOIN category c ON fc.category_id = c.category_id
    WHERE 
        c.name IN ('Animation', 'Children', 'Classics', 'Comedy', 'Family', 'Music')
)
SELECT
    title,
    name,
    rental_duration,
    quartile AS standard_quartile
FROM
    RentalDurationQuartiles


/*
Question 3:
Provide a table with the family-friendly film category, each of the quartiles, and the corresponding count of movies within each combination of film category for each corresponding rental duration category. The resulting table should have three columns:
    - Category
    - Rental length category
    - Count
*/

WITH CategoryFilmQuartiles AS (
    SELECT 
        c.name , 
        f.rental_duration,
        NTILE(4) OVER (ORDER BY f.rental_duration) AS standard_quartile
    FROM 
        category c
    JOIN 
        film_category fc ON c.category_id = fc.category_id
    JOIN 
        film f ON fc.film_id = f.film_id
    WHERE 
        c.name IN ('Animation', 'Children', 'Classics', 'Comedy', 'Family', 'Music')
)

SELECT 
    name, 
    standard_quartile,
    COUNT(*) AS count
FROM 
    CategoryFilmQuartiles
GROUP BY 
    name, standard_quartile
ORDER BY 
    name, standard_quartile;


----------------- SET 2 -------------------------

/*
Question 1: Count of rental orders during every month for all the years 
*/


SELECT 
  EXTRACT(YEAR FROM r.rental_date) AS Rental_Year, 
  EXTRACT(MONTH FROM r.rental_date) AS Rental_Month, 
  s.store_id Store_ID, 
  COUNT(*) AS Count_Rentals
FROM 
  store s
  INNER JOIN staff st ON s.store_id = st.store_id
  INNER JOIN rental r ON st.staff_id = r.staff_id
GROUP BY 
  Rental_Year, 
  Rental_Month, 
  s.store_id
ORDER BY 
  Count_Rentals DESC;


/*
Question 2: Query to capture the customer name, month and year of payment, and total payment amount for each month by these top 10 paying customers?
*/

WITH top10_customers AS (
    SELECT c.customer_id, SUM(p.amount) AS total_payments
    FROM customer c
    JOIN payment p ON p.customer_id = c.customer_id
    GROUP BY c.customer_id
    ORDER BY total_payments DESC
    LIMIT 10
)

SELECT DATE_TRUNC('month', p.payment_date) AS pay_month,
       CONCAT(c.first_name, ' ', c.last_name) AS full_name,
       COUNT(p.amount) AS pay_count_per_month,
       SUM(p.amount) AS pay_amount
FROM top10_customers t
JOIN customer c ON t.customer_id = c.customer_id
JOIN payment p ON p.customer_id = c.customer_id
WHERE p.payment_date >= '2007-01-01' AND p.payment_date < '2008-01-01'
GROUP BY 1, 2
ORDER BY full_name, pay_month;

/*
Question 3: Query to compare the payment amounts in each successive month
*/

WITH Top10Customers AS (
  SELECT c.customer_id
  FROM customer c
  JOIN payment p ON c.customer_id = p.customer_id
  GROUP BY c.customer_id
  ORDER BY SUM(p.amount) DESC
  LIMIT 10
),
MonthlyPayments AS (
  SELECT
    c.customer_id,
    TO_CHAR(p.payment_date, 'YYYY-MM') AS payment_month,
    (c.first_name || ' ' || c.last_name) AS full_name,
    SUM(p.amount) AS monthly_payment
  FROM payment p
  JOIN customer c ON p.customer_id = c.customer_id
  WHERE c.customer_id IN (SELECT customer_id FROM Top10Customers)
    AND p.payment_date >= '2007-01-01' AND p.payment_date < '2008-01-01'
  GROUP BY c.customer_id, payment_month, full_name
),
PaymentDifferences AS (
  SELECT
    customer_id,
    full_name,
    payment_month,
    monthly_payment,
    monthly_payment - COALESCE(LAG(monthly_payment) OVER (PARTITION BY customer_id ORDER BY payment_month), 0) AS payment_difference
  FROM MonthlyPayments
)
SELECT *
FROM PaymentDifferences
ORDER BY payment_difference DESC;