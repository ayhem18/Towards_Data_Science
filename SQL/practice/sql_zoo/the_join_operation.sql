-- THESE ARE MY SOLUTIONS FOR the join operation tutorial. The problems can be found through the following link: https://sqlzoo.net/wiki/The_JOIN_operation

-- PROBLEM 4
SELECT team1, team2, player
FROM game
JOIN goal ON game.id = goal.matchid
WHERE player LIKE 'Mario%';


-- PROBLEM 5
SELECT player, teamid, coach, gtime
FROM eteam
JOIN goal ON teamid = id
WHERE gtime <= 10;

-- more optimized queris
SELECT player, teamid, coach, gtime
from goal as g
JOIN eteam 
ON eteam.id = g.teamid and g.gtime <= 10; -- using the filtering condition at the level of the "ON" statement filters the tables before joining
-- using condition on the level of the where statement filters after joining 


-- PROBLEM 6
SELECT mdate as "match_date", teamname
FROM game
JOIN eteam ON team1 = eteam.id
WHERE coach = 'Fernando Santos'; 

-- PROBLEM 7
SELECT player 
FROM game
JOIN goal ON id = matchid
WHERE stadium = 'National Stadium, Warsaw'; 

-- PROBLEM 8 : it might not be straightforward that the names should be unique (as some players such as Mario Balotelli scored more than once)
SELECT DISTINCT(player)
FROM goal 
JOIN game ON game.id = matchid
WHERE (team1 = (SELECT id FROM eteam where LOWER(teamname) = 'germany') 
			or 
		team2 = (SELECT id FROM eteam where LOWER(teamname) = 'germany'))
 AND 
 (teamid != (SELECT id FROM eteam where LOWER(teamname) = 'germany'));
 

SELECT DISTINCT(player) 
from goal as go JOIN game as ga
ON (ga.team1 = (SELECT id from eteam where lower(teamname) = 'germany') or ga.team2 = (SELECT id from eteam where lower(teamname) = 'germany')) 
	and go.teamid <> (SELECT id from eteam where lower(teamname) = 'germany') 
    and go.matchid = ga.id;



-- PROBLEM 9
select teamname, count(teamid) as goals_scored
FROM eteam
JOIN goal ON teamid = eteam.id 
GROUP BY teamid, teamname;
  
-- PROBLEM 10 
SELECT stadium, COUNT(stadium) as "goals_scored"
FROM game
JOIN goal ON game.id = matchid
GROUP BY stadium;  
  
-- PROBLEM 11
SELECT game.id, game.mdate as match_data, COUNT(game.id) as goals_scored
FROM game
JOIN goal ON game.id = goal.matchid
WHERE team1 = (SELECT id FROM eteam WHERE lower(teamname) = 'poland') OR
team2 = (SELECT id FROM eteam WHERE lower(teamname) = 'poland')
GROUP BY game.id, game.mdate;


SELECT id, mdate, Count(matchid) from
(SELECT id, mdate
from game 
where team1 = "POL" or team2 = "POL") as pol_games
LEFT JOIN goal
on goal.matchid = id
GROUP BY id, mdate;



-- PROBLEM 12
SELECT game.id, mdate as match_date, COUNT(game.id) as "Germany's goals"
FROM goal
JOIN game ON game.id = goal.matchid
WHERE teamid = 'GER'
GROUP BY game.id, game.mdate;


SELECT ger_goal_count, id, mdate
from 
(SELECT matchid, COUNT(*) as ger_goal_count from goal where teamid = 'GER' GROUP BY matchid) as ger_goals
JOIN game
ON game.id = ger_goal.matchid; 



-- PROBLEM 13
SELECT mdate, team1, SUM(CASE WHEN goal.teamid = team1 THEN 1 ELSE 0 END) as 'score1', 
team2, SUM(CASE WHEN goal.teamid = team2 THEN 1 ELSE 0 END) as 'score2'
FROM game
LEFT JOIN goal ON goal.matchid = game.id 
GROUP BY game.id, mdate, team1, team2
ORDER BY CAST(mdate AS DATE), game.id, team1, team2;
 