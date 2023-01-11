-- This is my solution to the STATION problem WEATHE OBSERVATION STATION 20: https://www.hackerrank.com/challenges/weather-observation-station-20/problem

-- save the number of rows in the table
SET @number = (SELECT COUNT(long_w) FROM station);

-- regardless of the parity of the total number, a median is always the average of the element of order
-- ceil(n + 1 / 2) and ceil(n / 2)

-- the idea is to consider a subquary with the sorted element first
-- define a variable to keep the order of the rows:

SET @num_row = 1;

SELECT (@num_row:=@num_row + 1) AS row_num, 
SUM(CASE WHEN row_num = CEIL(number / 2) OR row_num = CEIL((number + 1) / 2) THEN 1 ELSE 0) / 2 AS median
FROM ( SELECT long_w FROM station ORDER BY long_w) as t;

