-- understand how many order_ids out there
SELECT COUNT(DISTINCT(order_id)) FROM order_props_value;

-- each of the statements below will return the same result
SELECT COUNT(order_id) FROM order_props_value where ORDER_PROPS_ID = 97;
SELECT COUNT(order_id) FROM order_props_value where ORDER_PROPS_ID = 95;
SELECT COUNT(order_id) FROM order_props_value where ORDER_PROPS_ID = 77;

-- we have 517610 unique orders in the order_props_value. The total number of records is 517610 * 6 (each order_id is repeated across 6 records with different ids)


-- we can except around 450k order_ids with valid start and finish prep time
select count(order_id) as "c", count(DISTINCT(order_id)) as "dc"
from order_props_value
where ORDER_PROPS_ID = 97 
and not value is Null;

select count(order_id) as "c", count(DISTINCT(order_id)) as "dc"
from order_props_value
where ORDER_PROPS_ID = 95; 
and not value is Null;

--however there are still many orders with extremely low preptime

SELECT * from order_props;


SELECT order_id 
from ORDER_PROPS_ID
where 

SELECT order_id 
from order_props_value
where ORDER_PROPS_ID = 18 and value is Null
limit 10;



SELECT *
from 
(SELECT *
	from order_props_value as o1 
	JOIN order_props_value as o2  
	ON o1.ORDER_PROPS_ID = 95 and (not o1.value is null) -- choose a finish_prep_date that is not null
	and o2.ORDER_PROPS_ID= 97 and (not o2.value is Null) -- choose a start_prep_date that is not null
	and o1.order_id = o2.order_id
	where o1.value = o2.value
) as e
join ORDER_PROPS_value as o
ON o.order_id = e.order_id







-- final query

SELECT orders.order_id, 

-- ordering the start and finish prep dates
(CASE orders.start_prep_date <= orders.finish_prep_date WHEN 1 THEN orders.start_prep_date ELSE orders.finish_prep_date end) as "start_prep_date", 

(CASE orders.finish_prep_date <= orders.start_prep_date WHEN 1 THEN orders.start_prep_date ELSE orders.finish_prep_date end) as "finish_prep_date",

orders.profit, 
orders.delivery_distance, 
orders.region_id,

oh.DATE_CREATE as "order_created_date",

oh.STATUS_ID, 
oh.planned_prep_time, 
ob.product_id, 
ob.store_id, 
ob.price

FROM (
	SELECT o1.order_id, o2.value as "start_prep_date", o1.value as "finish_prep_date", o3.value as "profit", o4.value as "delivery_distance", o5.value as "region_id"

	from order_props_value as o1 

	JOIN order_props_value as o2  
	ON o1.ORDER_PROPS_ID = 95 and (not o1.value is null) -- choose a finish_prep_date that is not null
	and o2.ORDER_PROPS_ID= 97 and (not o2.value is Null) -- choose a start_prep_date that is not null
	and o1.order_id = o2.order_id

	JOIN order_props_value as o3
	ON o3.ORDER_PROPS_ID = 77 and o1.ORDER_ID = O3.order_id

	JOIN order_props_value as o4
	ON o4.ORDER_PROPS_ID = 65 AND O1.order_id = o4.order_id
	
	JOIN order_props_value as o5 
	ON o5.ORDER_PROPS_ID = 11 AND o1.order_id = o5.order_id
) as orders

JOIN order_history as oh
ON oh.order_id = orders.order_id

JOIN (SELECT store_id, product_id, order_id, price from order_busket) as ob
on ob.order_id = orders.order_id

