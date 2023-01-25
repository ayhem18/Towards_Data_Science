-- THESE ARE MY SOLUTIONS FOR THE AGGREGATION TUTOIRAL ON SQL_ZOO. the problems can be found throug the link: https://sqlzoo.net/wiki/SUM_and_COUNT 

-- PROBLEM 4
SELECT COUNT(area) as big_countries_count
FROM world
WHERE area >= 1000000;

-- PROBLEM 5 
SELECT SUM(population) as baltic_population
FROM world 
WHERE name IN  ('Estonia', 'Latvia', 'Lithuania');

-- PROBLEM 6
SELECT continent, count(name) as countries_per_contient
FROM world
GROUP BY continent;

-- PROBLEM 7
SELECT continent, count(name) as countries_per_contient
FROM world
WHERE population >= 10000000
GROUP BY continent;

-- PROBLEM 8
SELECT continent
FROM world
GROUP BY continent
HAVING SUM(population) >= 100000000;




