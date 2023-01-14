-- This is my solution to the STATION problem WEATHER OBSERVATION STATION 20: https://www.hackerrank.com/challenges/weather-observation-station-20/problem

-- save the number of rows in the table
SET @num = (SELECT COUNT(lat_n) FROM station);

-- regardless of the parity of the total number, a median is always the average of the element of order
-- ceil(n + 1 / 2) and ceil(n / 2)

-- the idea is to consider a subquary with the sorted elements first
-- define a variable to keep the order of the rows:

SET @num_row = 0;
DROP TABLE IF EXISTS temp_table ;
CREATE TABLE IF NOT EXISTS temp_table AS (SELECT t1.lat_n, (@num_row:=@num_row+ 1) AS row_num 
FROM 
(SELECT lat_n FROM  station ORDER BY lat_n) AS t1 
);

-- SELECT * FROM temp_table;

SELECT ROUND(SUM(CASE WHEN row_num = CEIL(@num/ 2) OR row_num  = CEIL((@num + 1) / 2) THEN lat_n ELSE 0 END) 
/ (CEIL((@num + 1) / 2) - CEIL(@num/ 2)  + 1), 4) AS median 
FROM temp_table;

-- solution for the following problem: 


