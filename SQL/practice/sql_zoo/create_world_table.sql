-- this script was written to mimic the 'world' table used in the sql zoo sql problem set.
-- some of the problems require some lengthy queries which can be tedious to write directly on the site (so local experimentation is crucial)

DROP DATABASE IF EXISTS test_db;
CREATE DATABASE test_db;
USE `test_db`;


CREATE TABLE IF NOT EXISTS world (
	name VARCHAR(50) NOT NULL,
	continent VARCHAR(60) NOT NULL,
	area DECIMAL(10, 0),
	population DECIMAL(11, 0),
	gdp DECIMAL(14, 0), 
	capital VARCHAR(60),
	PRIMARY KEY(name)
);

INSERT INTO world 
(name, continent, area, population, gdp, capital)
VALUES
("Afghanistan", "Asia", 652230, 25500100, 20364000000, "Kabul"), 
("Albania", "Europe", 28748, 2821977, 12044000000, "Tirana"), 
("Algeria", "Africa", 2381741, 38700000,207021000000, "Algiers"), 
("Andorra", "Europe", 468, 76098, 3222000000, "Andorra la Vella"), 
("Angola", "Africa", 1246700, 19183590, 116308000000, "Luanda"), 
("Antigua and Barbuda", "Caribbean", 442, 86295, 1176000000, "St. John's"),
("Germany",	"Europe",	357114, 80716000, 3425956000000, "Berlin");
