-- PROBLEM 
USE `test_db`;

/*
SELECT name, 
CONCAT(
	CONVERT(
		ROUND(
			  (population * 100) / 
			  (SELECT population from world where lower(name) = 'germany')
			 ), 
	 CHAR(3)), 
"%") 
as 'percentage' 

FROM world

where lower(continent) = 'europe';
*/

SELECT continent, name, area from world x
where area = (SELECT MAX(area) from world y where y.continent = x.continent);

SELECT continent, min(name) from world group by continent; 


