-- my solution for https://sqlpad.io/questions/19/most-productive-actor/

SELECT first_name, last_name -- the final columns to send to the judge
FROM actor 
WHERE actor_id IN 
(SELECT actor_id FROM  -- this select statement is used mainly before IN with a subquery requires the latter to have only one column
	   -- this subquery first row of the table where each actor_id is associated with the number of films that actor appears in
       -- order in descending order by the number of movies.
       (SELECT film_actor.actor_id, COUNT(film_actor.actor_id) as actor_cnt
		FROM actor
		JOIN film_actor ON film_actor.actor_id = actor.actor_id
		GROUP BY film_actor.actor_id
		ORDER BY actor_cnt DESC -- descending order
		LIMIT 1 ) as t1); -- only need the actor with the most appearances

-- my solution for: https://sqlpad.io/questions/20/customer-who-spent-the-most/

SELECT first_name, last_name
FROM customer
WHERE customer_id IN 
	(SELECT customer_id FROM 
	 (SELECT customer.customer_id, SUM(payment.amount) as spending
	  	FROM customer
	  	JOIN payment ON customer.customer_id = payment.customer_id
	  	-- filter by the time period specified
		WHERE payment_ts >= '2020-02-01' AND payment_ts < '2020-03-01' -- filter by Februray 2020
	  	GROUP BY customer.customer_id
	  	ORDER BY spending DESC 
	  	LIMIT 1) as t1); 
        
-- my solution for: https://sqlpad.io/questions/21/customer-who-rented-the-most/

SELECT first_name, last_name
FROM 
customer
JOIN 
(SELECT customer.customer_id, COUNT(customer.customer_id) as rentals
   FROM customer
   JOIN rental ON customer.customer_id = rental.customer_id
   -- filter by the time period specified
   WHERE rental_ts >= '2020-05-01' AND rental_ts < '2020-06-01' -- filter by May 2020
   GROUP BY customer.customer_id
   ORDER BY rentals DESC 
   LIMIT 1) as t1
ON customer.customer_id = t1.customer_id;


-- my solution for: https://sqlpad.io/questions/26/second-shortest-film/

SELECT title
FROM film
WHERE length != (SELECT MIN(length) FROM film)
ORDER BY length
LIMIT 1;


-- my solution for: https://sqlpad.io/questions/27/film-with-the-largest-cast/

SELECT title 
FROM (SELECT film.film_id, COUNT(film.film_id) as film_cast
	 FROm film
	 JOIN film_actor ON film.film_id = film_actor.film_id
	 GROUP BY film.film_id 
	 ORDER BY film_cast DESC
	 LIMIT 1) as t -- the table t is only the first row of table temp where each film_id is associated with the number of actors working on it
     -- ordered by the number of actors in descending order.
JOIN film ON film.film_id = t.film_id;

-- the problem: https://sqlpad.io/questions/28/film-with-the-second-largest-cast/ is a more sophisticated version of the problem above.
