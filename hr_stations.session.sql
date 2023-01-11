SELECT * FROM station; 
SELECT MAX(lat_n) - MIN(lat_n) + MAX(long_w) - MIN(long_w) as distance
FROM station; 


SELECT ROUND(SQRT(POWER(MAX(lat_n) - MIN(lat_n), 2) +
 POWER(MAX(long_w) - MIN(long_w), 2)), 4) as distance
FROM station; 

