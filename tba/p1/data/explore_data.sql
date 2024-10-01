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
			SELECT order_id, max(value) as "finish_prep_date",  min(value) as "start_prep_date", 
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


-- each order has all the prop information needed
SELECT MAX(count) as "max_order_instances", MIN(count) as "min_order_instances" from 
(
	SELECT order_id, COUNT(*) as "count" 
	from order_props_value
	GROUP BY order_id
	
)as e

SELECT o1.order_id, o1.start_prep_date, o1.finish_prep_date, o2.value as "profit"
from (
	SELECT o1.order_id, o2.value as "start_prep_date", o1.value as "finish_prep_date"
	from order_props_value as o1 
	JOIN order_props_value as o2 
	ON o1.ORDER_PROPS_ID = 95 and o2.ORDER_PROPS_ID= 97 and o1.order_id = o2.order_id
) as o1

JOIN order_props_value as o2
ON o2.ORDER_PROPS_ID = 77 and o1.order_id = o2.order_id


-- just to make sure the reasoning and the code is correct
SELECT COUNT(DISTINCT(order_id)) FROM order_props_value;
SELECT COUNT(order_id) from 
(
	SELECT o1.order_id, o2.value as "start_prep_date", o1.value as "finish_prep_date", o3.value as "profit", o4.value as "delivery_distance"
	from order_props_value as o1 

	JOIN order_props_value as o2  
	ON o1.ORDER_PROPS_ID = 95 and o2.ORDER_PROPS_ID= 97 and o1.order_id = o2.order_id

	JOIN order_props_value as o3
	ON o3.ORDER_PROPS_ID = 77 and o1.ORDER_ID = O3.order_id

	JOIN order_props_value as o4
	ON o4.ORDER_PROPS_ID = 65 AND O1.order_id = o4.order_id
)
as orders


-- let's inspect the order_history a bit more
SELECT max(status_id), min(status_id) from order_history;

SELECT SUM(CASE STATUS_ID When 'F' THEN 1 ELSE 0 END ) as "f_count", 
SUM(CASE STATUS_ID When 'C' THEN 1 ELSE 0 END ) as "c_count" 
from order_history;

SELECT COUNT(order_id) from order_history;







