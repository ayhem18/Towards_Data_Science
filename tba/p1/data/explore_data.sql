-- let's see how tehe 'products' and 'order_busket' tables relate to each other

SELECT AVG(CASE ob_sid WHEN prod_sid THEN 1 else 0 END) as "store_match",
AVG(CASE ob_prod_id WHEN prod_prod_id THEN 1 else 0 END) as "product_id_match"

FROM 	
(
	SELECT ob.id as "ob_id", 
	p.id as "prod_instance_id", 
	ob.product_id as "ob_prod_id", 
	p.product_id as "prod_prod_id",
	ob.store_id as "ob_sid", p.store_id as "prod_sid"
	from order_busket as ob
	JOIN products as p 
	on ob.id = p.id
) as temp;
	

--  finding the start and finish prep times
SELECT order_id, max(value) as "finish_prep_date",  min(value) as "start_prep_date"
from (
	SELECT * from order_props_value 
	where ORDER_PROPS_ID IN (97, 95)
)
GROUP BY order_id
ORDER BY order_id
LIMIT 100;


-- join the 'order_history' table with the 'order_busket' 

SELECT 
e.order_id, e.finish_prep_date, e.start_prep_date, e.planned_prep_time, e.store_id, e.product_id, e.price, p.date_create as "product_creation_date"

FROM (
	SELECT 
	e.order_id, finish_prep_date, start_prep_date, planned_prep_time, store_id, product_id, price

	FROM 
	(
		SELECT oh.order_id, finish_prep_date, start_prep_date, planned_prep_time
		from order_history as oh
		join 
		(
			SELECT order_id, max(value) as "finish_prep_date",  min(value) as "start_prep_date"
			from 
				(
					SELECT * from order_props_value 
					where ORDER_PROPS_ID IN (97, 95)
				)
			GROUP BY order_id
		) as order_times
		
		on oh.order_id = order_times.order_id
		
	) as e

	JOIN order_busket as ob
	on ob.order_id = e.order_id
) as e

LEFT JOIN products as p

ON e.product_id = p.product_id;


SELECT * from order_props LIMIT 10;
SELECT * from order_props_value LIMIT 10;







