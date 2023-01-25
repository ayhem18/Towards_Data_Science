-- this is my solutions for SQL ZOO SELECT FROM WORLD. THE PROBLEMS CAN BE FOUND THROUGH THE LINK: https://sqlzoo.net/wiki/SELECT_from_WORLD_Tutorial

-- PROBLEM 3
SELECT name, GDP / population 
FROM world
where population > 200000000;

-- PROBLEM 4
SELECT name, population / 1000000 AS "population_in_millions"
FROM world
WHERE LOWER(continent) = 'south america';

-- PROBLEM 5
SELECT name, population 
FROM world
WHERE LOWER(name) IN ('france', 'germany', 'italy');

-- PROBLEM 6
SELECT name 
FROM world
WHERE LOWER(name) LIKE'%united%';

-- PROBLEM 7
SELECT name, population, area
FROM world
WHERE population > 250000000 OR area > 3000000;

-- PROBLEM 8
SELECT name, population, area
FROM world
WHERE (population > 250000000) != (area > 3000000);

-- PROBLEM 9
SELECT name, ROUND(population / 1000000, 2) as "population_in_millions", ROUND(gdp / 1000000000, 2) AS "gdp_in_millions" 
FROM world
WHERE LOWER(continent) = 'south america';
 
 -- PROBLEM 10
SELECT name, ROUND(gdp / (1000 * population)) * 1000 as "gdp_per_capita"
FROM world
WHERE gdp > 1000000000000;

-- PROBLEM 11
SELECT name, capital
FROM world
WHERE LENGTH(capital) = LENGTH(name);

-- PROBLEM 12
SELECT name, capital
FROM world 
WHERE name != capital AND LEFT(name, 1) = LEFT(capital, 1);

-- PROBLEM 13
SELECT name
FROM world
WHERE name LIKE '%a%' and 
      name LIKE '%e%' and
      name LIKE '%u%' and
      name LIKE '%i%' and	
	  name LIKE '%o%' and
      name NOT LIKE '% %'; 







