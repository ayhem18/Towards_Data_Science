-- These are my solutions for the SELECT WITHIN SELECT TUTORIAL. THE PROBLEMS CAN BE FOUND THROUGH THIS LINK: https://sqlzoo.net/wiki/SELECT_within_SELECT_Tutorial 

-- PROBLEM 3
SELECT name, continent
FROM world
WHERE continent IN (
    SELECT continent 
    FROM world 
    WHERE lower(name) IN ('argentina', 'australia'))
ORDER BY name;

-- PROBLEM 4
SELECT name, population
FROM world
WHERE 
(SELECT population FROM world WHERE LOWER(name) = 'united kingdom') < population AND population < (SELECT population FROM world WHERE LOWER(name) = 'germany');


-- PROBLEM 5
SELECT name as Name, 
CONCAT(
ROUND(
(population * 100)/ (SELECT population FROM world WHERE lower(name) ='germany')),
'%')
as "Percentage"
FROM world
WHERE LOWER(continent) = 'europe'; 

-- PROBLEM 6

SELECT name from world
where GDP > 
(SELECT MAX(GDP) from world where lower(continent) = 'europe');


-- PROBLEM 7
SELECT continent, name, area from world x
where area = (SELECT MAX(area) from world y where y.continent = x.continent);

-- PROBLEM 8
SELECT continent, min(name) from world group by continent;

-- PROBLEM 9: I think hardest part is understanding the problem formulation; it can be formulated as: find countries belonging to continents where the maximum population is at most 25mil
SELECT name, continent, population 
from world
where continent in 
(SELECT continent from world group by continent 
HAVING MAX(population) <= 25000000);
