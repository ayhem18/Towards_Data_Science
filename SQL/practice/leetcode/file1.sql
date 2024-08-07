

-- https://leetcode.com/problems/managers-with-at-least-5-direct-reports/description/

SELECT name 
FROM (
    SELECT managerId as id , COUNT(managerId) as report_count
    FROM  Employee
    GROUP BY managerId 
    HAVING COUNT(managerId) >= 5 
) as m
JOIN Employee as e 
ON e.id = m.id;

-- https://leetcode.com/problems/product-sales-analysis-iii/
SELECT product_id, year, quantity, price
FROM 
( SELECT *, MIN(year) OVER (PARTITION BY product_id)  as first_year FROM sales) as s 
WHERE  s.yr = s.first_year;

--- sol2
SELECT product_id, first_year, quantity, price 
FROM (
	SELECT product_id, min(year) as first_year
	from sales
	GROUP BY product_id
) AS sales_per_product
JOIN sales
ON sales_per_product.product_id = sales.product_id and sales_per_product.first_year = sales.year;
 

-- https://leetcode.com/problems/movie-rating/
(SELECT name as results 
    from 
	(
		SELECT user_id, COUNT(user_id) OVER(PARTITION BY user_id) as rate_avg
		FROM MovieRating
	) as u 

JOIN Users 
ON u.user_id = Users.user_id

ORDER BY rate_avg DESC, name
LIMIT 1
)

UNION ALL -- for some reason they have a test case where the name of the movie and the name of the user are the same 

(SELECT title as results from 
	(
        SELECT movie_id, AVG(rating) OVER (PARTITION BY movie_id) as avg_rate 
        FROM MovieRating
        WHERE YEAR(created_at) = 2020 and MONTH(created_at) = 2
	) as r
JOIN Movies 
ON Movies.movie_id = r.movie_id
ORDER BY avg_rate DESC, title
LIMIT 1
)
