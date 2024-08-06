-- THESE ARE MY SOLUTION FOR THE MORE JOIN OPERATIONS TUTORIAL ON SQL_ZOO SITE, THE PROBLEMS CAN BE FOUND THROUGH THIS LINK: https://sqlzoo.net/wiki/More_JOIN_operations
-- ONLY PROBLEMS USING JOIN ARE INCLUDED IN THE SCRIPT, as the first introductory problems are relatively simple

-- PROBLEM 6
SELECT actor.name
FROM movie, actor, casting
WHERE movie.id = (SELECT id FROM movie WHERE title = 'Casablanca')
AND movie.id = casting.movieid AND casting.actorid = actor.id;

-- PROBLEM 7
SELECT actor.name
FROM movie, actor, casting
WHERE movie.id = (SELECT id FROM movie WHERE title = 'Alien')
AND movie.id = casting.movieid AND casting.actorid = actor.id;

-- PROBLEM 8
SELECT title 
FROM movie
JOIN (
	SELECT movieid 
	FROM casting 
	where actorid = (SELECT id from actor where lower(name) = 'Harrison Ford')
) as ford_movies 
ON movie.id = ford_movies.movieid;


-- PROBLEM 9
SELECT title 
FROM movie
JOIN (
	SELECT movieid 
	FROM casting 
	where actorid = (SELECT id from actor where lower(name) = 'Harrison Ford') and ord <> 1
) as ford_movies 
ON movie.id = ford_movies.movieid;


-- PROBLEM 10
SELECT title, actor.name
FROM (
	SELECT movieid, title, actorid from movie
	JOIN casting
	ON movie.yr = 1962 and casting.ord = 1 and casting.movieid=movie.id
) as a

JOIN actor
ON a.actorid = actor.id;


-- PROBLEM 11
SELECT yr, COUNT(yr) as movie_year_count 
FROM movie
JOIN casting
ON 
	(actorid = (SELECT id from actor where LOWER(name) = 'Rock Hudson') 
	and movie.id = casting.movieid
	)
GROUP BY yr
HAVING COUNT(yr) > 2;


-- PROBLEM 12 
SELECT DISTINCT(title), leading_actor
FROM 
(SELECT title, movieid 
FROM movie
JOIN casting
ON 
	(actorid = (SELECT id from actor where LOWER(name) = 'Julie Andrews') 
	and movie.id = casting.movieid
	) 
-- this table finds the title and ids of movies in which Julie Andrews tookpart
) as m

JOIN 
(SELECT actorid, name as leading_actor, movieid
FROM casting 
JOIN actor
ON casting.ord = 1 and casting.actorid = actor.id) as a

ON m.movieid = a.movieid;


-- PROBLEM 13
SELECT name 
FROM (
	SELECT actorid FROM casting 
	where ord = 1
    GROUP BY actorid 
    HAVING COUNT(actorid) >= 15
) as a
 
JOIN actor
ON a.actorid = actor.id
ORDER BY name;



-- PROBLEM 14
SELECT title, cast_size
FROM (
	SELECT movieid, COUNT(movieid) as cast_size
	FROM casting 
	GROUP BY movieid
) 
as movie_cast_size
JOIN movie
ON yr = 1978 and movie.id = movie_cast_size.movieid
ORDER BY cast_size DESC, title ;


-- PROBLEM 15

--first select the movies in which Art Garfunkel took part
SELECT name
FROM 
(
    SELECT c2.actorid
    from casting as c1
    JOIN casting as c2 
    ON 
    c1.actorid = (SELECT id FROM actor where name = "Art Garfunkel") 
    and 
    (c1.movieid = c2.movieid and c2.actorid <> (SELECT id FROM actor where name = "Art Garfunkel"))
) as a

JOIN actor
ON a.actorid = actor.id;

