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


-- PROBLEM 6
SELECT name as Name, 
CONCAT(
ROUND(
(population * 100)/ (SELECT population FROM world WHERE lower(name) ='germany')),
'%')
as "Percentage"
FROM world
WHERE LOWER(continent) = 'europe'; 
