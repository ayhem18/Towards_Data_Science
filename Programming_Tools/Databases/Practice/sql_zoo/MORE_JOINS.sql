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
