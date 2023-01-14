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


-- PROBLEM 12
SELECT game.id, mdate as match_date, COUNT(game.id) as "Germany's goals"
FROM goal
JOIN game ON game.id = goal.matchid
WHERE teamid = 'GER'
GROUP BY game.id, game.mdate;

-- PROBLEM 13
SELECT mdate, team1, SUM(CASE WHEN goal.teamid = team1 THEN 1 ELSE 0 END) as 'score1', 
team2, SUM(CASE WHEN goal.teamid = team2 THEN 1 ELSE 0 END) as 'score2'
FROM game
LEFT JOIN goal ON goal.matchid = game.id 
GROUP BY game.id, mdate, team1, team2
ORDER BY CAST(mdate AS DATE), game.id, team1, team2;
 