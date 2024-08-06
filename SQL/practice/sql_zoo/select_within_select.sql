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

SELECT DISTINCT(player) 
from goal as go JOIN game as ga
ON (ga.team1 = (SELECT id from eteam where lower(teamname) = 'germany') or ga.team2 = (SELECT id from eteam where lower(teamname) = 'germany')) 
	and go.teamid <> (SELECT id from eteam where lower(teamname) = 'germany') 
    and go.matchid = ga.id;
    
SELECT teamname, goal_count
from 
(SELECT teamid, count(*) as goal_count from goal GROUP BY teamid) as goal_count
JOIN eteam
on goal_count.teamid = eteam.id;

SELECT teamname, COUNT(*)
from eteam
LEFT join goal
ON eteam.id = goal.teamid
GROUP BY teamid ;

SELECT stadium, COUNT(*) as goal_count
from game
LEFT JOIN goal
on game.id = goal.matchid
GROUP BY stadium;

SELECT id, mdate, Count(matchid) from
(SELECT id, mdate
from game 
where team1 = "POL" or team2 = "POL") as pol_games
LEFT JOIN goal
on goal.matchid = id
GROUP BY id, mdate;

SELECT ger_goal_count, id, mdate
from 
(SELECT matchid, COUNT(*) as ger_goal_count from goal where teamid = 'GER' GROUP BY matchid) as ger_goals
JOIN game
ON game.id = ger_goal.matchid; 

SELECT name FROM actor
JOIN 
(SELECT actorid as id from casting 
where movieid = (SELECT id from movie where title='Alien')
) as alien_actors
ON alien_actors.id = actor.id;

SELECT title 
FROM movie
JOIN (
	SELECT movieid 
	FROM casting 
	where actorid = (SELECT id from actor where lower(name) = 'Harrison Ford')
) as ford_movies 
ON movie.id = ford_movies.movieid;

SELECT title 
FROM movie
JOIN (
	SELECT movieid 
	FROM casting 
	where actorid = (SELECT id from actor where lower(name) = 'Harrison Ford') and ord <> 2
) as ford_movies 
ON movie.id = ford_movies.movieid;


SELECT title, actor.name
FROM (
	SELECT movieid, title, actorid from movie
	JOIN casting
	ON movie.yr = 1962 and casting.ord = 1 and casting.movieid=movie.id
) as a

JOIN actor
ON a.actorid = actor.id;

SELECT yr, COUNT(yr) as movie_year_count 
FROM movie
where actorid = (SELECT id from actor where LOWER(name) = 'Rock Hudson')
GROUP BY yr 
HAVING COUNT(yr) > 2;


SELECT DISTINCT(title, leading_actor)
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

SELECT name 
FROM (
	SELECT actorid FROM casting 
	where ord = 1
    GROUP BY actorid 
    HAVING COUNT(actorid) >= 15
) as a
 
JOIN actor
ON a.actorid = actor.id;

SELECT title
FROM (
	SELECT movieid, COUNT(movieid) as cast_size
	FROM casting 
	GROUP BY movieid
) 
as movie_cast_size
JOIN movie
ON yr = 1978 and movie.id = movie_cast_size.movieid
ORDER BY cast_size, title ;


