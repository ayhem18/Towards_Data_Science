-- this is my solution to the HACKER RANK CHALLENGE: https://www.hackerrank.com/challenges/challenges/problem?isFullScreen=true 

-- first let's join our table and extract the number of challenges each challenger created
-- and save the table for later use

DROP TABLE IF EXISTS challenge_count;
CREATE TABLE IF NOT EXISTS  challenge_count 
AS (SELECT hackers.hacker_id, COUNT(hackers.hacker_id) as challenge_count
FROM hackers
JOIN challenges ON hackers.hacker_id = challenges.hacker_id
GROUP BY  hackers.hacker_id
ORDER By challenge_count DESC
);

-- SELECT * FROM challenge_count;

-- let's extrct the maximum number of challenges created by a single hacker
 SET @max_count = (SELECT MAX(challenge_count) FROM challenge_count);

-- we need to sort the students by the number of challenges they created
-- and then hacker_id. Students that share the same number of challenges while this number is not the maximum are droppped
-- This can be done as follows:

-- the table commented below extracts the count of the count of challenges created:
-- the dropped column determines whether a certain name should be included in the final result or not.

/*
SELECT challenge_count, COUNT(challenge_count) AS challenge_count_count,
 (CASE WHEN challenge_count != @max_count AND COUNT(challenge_count) > 1 THEN 1 ELSE 0 END) as dropped
FROM challenge_count
GROUP BY challenge_count;
*/

-- the next step is to join the challenge_count table with hackers table and the table above
SELECT challenge_count.hacker_id, hackers.name, challenge_count.challenge_count
FROM hackers, challenge_count, 

(SELECT challenge_count, COUNT(challenge_count) AS challenge_count_count,
 (CASE WHEN challenge_count != @max_count AND COUNT(challenge_count) > 1 THEN 1 ELSE 0 END) as dropped
FROM challenge_count
GROUP BY challenge_count) as t

WHERE t.dropped != 1 AND hackers.hacker_id = challenge_count.hacker_id AND challenge_count.challenge_count = t.challenge_count
ORDER BY challenge_count DESC, hacker_id ASC;


