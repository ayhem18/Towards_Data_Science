-- THIS SQL SCRIPT IS USED TO CREATE THE STATION TABLE USED IN WHETHER OBSERVATION FAMILY OF PROBLEMS

CREATE TABLE IF NOT EXISTS STATION (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    city VARCHAR(21), 
    state VARCHAR(2),
    lat_n SMALLINT,
    long_w SMALLINT
);
-- THIS SQL SCRIPT IS USED TO CREATE THE STATION TABLE USED IN WHETHER OBSERVATION FAMILY OF PROBLEMS

CREATE TABLE IF NOT EXISTS STATION (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    city VARCHAR(21), 
    state VARCHAR(2),
    lat_n SMALLINT,
    long_w SMALLINT
);

SELECT * FROM station;

-- populate the table with some values for local testing
INSERT INTO STATION (city, state, lat_n, long_w)    
    VALUES ('aa', 'A', 11, 1),
     ('bb', 'B', 22, 11),
     ('cc', 'A', 33, 31),
     ('dd', 'D', 44, 21),
     ('ad', 'C', 5,19),
     ('ac', 'C', 79, 21),
     ('aba', 'B', 79, 31),
     ('baa', 'B', 69, 41),
     ('cd', 'A', 59, 51),
     ('da', 'A', 49, 61),
     ('aa', 'B', 39, 71),
     ('aaa', 'C', 29, 90),
     ('aaa', 'D', 1, 110);

SELECT * FROM STATION;

SELECT * FROM station;

-- populate the table with some values for local testing
INSERT INTO STATION (city, state, lat_n, long_w)    
    VALUES ('aa', 'A', 11, 1),
     ('bb', 'B', 22, 11),
     ('cc', 'A', 33, 31),
     ('dd', 'D', 44, 21),
     ('ad', 'C', 5,19),
     ('ac', 'C', 79, 21),
     ('aba', 'B', 79, 31),
     ('baa', 'B', 69, 41),
     ('cd', 'A', 59, 51),
     ('da', 'A', 49, 61),
     ('aa', 'B', 39, 71),
     ('aaa', 'C', 29, 90),
     ('aaa', 'D', 1, 110);

SELECT * FROM STATION;
SELECT * FROM station;

-- populate the table with some values for local testing
INSERT INTO STATION (city, state, lat_n, long_w)    
    VALUES ('aa', 'A', 11, 1),
     ('bb', 'B', 22, 11),
     ('cc', 'A', 33, 31),
     ('dd', 'D', 44, 21),
     ('ad', 'C', 5,19),
     ('ac', 'C', 79, 21),
     ('aba', 'B', 79, 31),
     ('baa', 'B', 69, 41),
     ('cd', 'A', 59, 51),
     ('da', 'A', 49, 61),
     ('aa', 'B', 39, 71),
     ('aaa', 'C', 29, 90),
     ('aaa', 'D', 1, 110);

SELECT * FROM STATION;