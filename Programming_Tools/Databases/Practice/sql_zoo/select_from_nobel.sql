-- these are my solutions for the SELECT FROM NOBEL TUTORIAL ON SQL_ZOO. the problems can be found through this link: https://sqlzoo.net/wiki/SELECT_from_Nobel_Tutorial


-- PROBLEM 3
SELECT yr, subject
FROM nobel
WHERE LOWER(winner) = 'albert einstein';  

-- PROBLEM 4
SELECT name
FROM nobel 
where subject = 'peace' and yr >= 2000;

-- PROBLEM 5
SELECT yr, subject, winner 
FROM nobel
WHERE (yr BETWEEN 1980 AND 1989) AND subject= 'literature';

-- PROBLEM 6
SELECT *
FROM nobel
WHERE winner IN ('Theodore Roosevelt', 'Woodrow Wilson', 'Jimmy Carter', 'Barack Obama'); 

-- PROBLEM 7 
SELECT winner
FROM nobel
WHERE LOWER(winner) LIKE 'john%';

-- PROBLEM 8
SELECT yr, subject, winner
FROM nobel
WHERE (yr = 1980 and subject = 'physics') OR 
(yr = 1984 and subject = 'chemistry'); 

-- PROBLEM 9
SELECT *
FROM nobel
WHERE (yr = 1980) and subject not in ('chemistry', 'medicine');

-- PROBLEM 10
SELECT * 
FROM nobel
WHERE (yr < 1910 AND subject = 'Medicine') OR (yr >= 2004 AND subject = 'Literature');

-- PROBLEM 11
SELECT * 
FROM nobel
WHERE winner = 'PETER GRÜNBERG'; -- just copy past the letter from the link attached to the problem

-- PROBLEM 12
SELECT * 
FROM nobel
WHERE winner = 'EUGENE O\'NEILL';


-- PROBLEM 13
SELECT winner, yr, subject
FROM nobel
WHERE LOWER (winner) LIKE 'sir%'
ORDER BY yr DESC, winner; 

-- PROBLEM 14
SELECT winner, subject
FROM nobel WHERE yr = 1984
ORDER BY subject IN ('physics', 'chemistry'), subject, winner;



