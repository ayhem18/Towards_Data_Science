/*
*********************************************************************
http://www.mysqltutorial.org
*********************************************************************
Name: MySQL Sample Database for Python
Link: http://www.mysqltutorial.org/
Version 1.0
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`python_mysql` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `python_mysql`;

/*Table structure for table `authors` */

DROP TABLE IF EXISTS `authors`;

CREATE TABLE `authors` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `first_name` varchar(40) NOT NULL,
  `last_name` varchar(40) NOT NULL,
  `photo` blob,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=128 DEFAULT CHARSET=latin1;

/*Data for the table `authors` */

insert  into `authors`(`id`,`first_name`,`last_name`,`photo`) values (1,'Herbert','Methley ',NULL),(2,'Olive','Wellwood ',NULL),(3,'Francis','Tugwell ',NULL),(4,'Randolph','Henry Ash',NULL),(5,'Christabel','LaMotte ',NULL),(7,'Robert','Dale Owen',NULL),(12,'Leonora','Stern ',NULL),(14,'Mrs.','Lees ',NULL),(19,'Josephine','M. Bettany',NULL),(20,'Kester','Bellever ',NULL),(22,'Sylvia','Leigh ',NULL),(24,'Elizabeth','Temple ',NULL),(25,'Marsha','Patterson ',NULL),(26,'Samuel','Humber ',NULL),(27,'James','Fallon ',NULL),(28,'Beatrice','Quinn ',NULL),(29,'Susan','Magar ',NULL),(30,'Clinton','York ',NULL),(31,'Canton','Lee ',NULL),(32,'Rod','Keen ',NULL),(33,'Hilda','Simpson ',NULL),(34,'S.','M. Justice',NULL),(35,'Charles','Green ',NULL),(36,'Richard','Brautigan ',NULL),(37,'Bill','Lewis ',NULL),(38,'Chuck',' ',NULL),(39,'Doctor','O. ',NULL),(40,'Harlow','Blade ',NULL),(41,'Barbara','Jones ',NULL),(42,'Fred','Sinkus ',NULL),(43,'Thomas','Funnel ',NULL),(44,'Patricia','Evens Summers',NULL),(45,'Reverend','Lincoln Lincoln',NULL),(47,'Edward','Fox ',NULL);

/*Table structure for table `book_author` */

DROP TABLE IF EXISTS `book_author`;

CREATE TABLE `book_author` (
  `book_id` int(11) NOT NULL,
  `author_id` int(11) NOT NULL,
  KEY `book_id` (`book_id`),
  KEY `author_id` (`author_id`),
  CONSTRAINT `ba_fk1` FOREIGN KEY (`book_id`) REFERENCES `books` (`id`) ON DELETE CASCADE,
  CONSTRAINT `ba_fk2` FOREIGN KEY (`author_id`) REFERENCES `authors` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `book_author` */

insert  into `book_author`(`book_id`,`author_id`) values (1,1),(2,1),(3,1),(4,1),(5,1),(6,2),(7,2),(8,2),(9,2),(10,2),(11,3),(12,4),(13,4),(14,4),(15,5),(16,4),(17,7),(18,5),(19,4),(20,4),(21,4),(22,4),(23,4),(24,5),(25,5),(26,4),(27,12),(28,4),(29,4),(30,14),(31,4),(32,4),(33,5),(34,5),(35,1),(36,1),(37,1),(38,1),(39,1),(40,2),(41,2),(42,2),(43,2),(44,2),(45,19),(46,19),(47,19),(48,19),(49,19),(50,19),(51,19),(52,20),(53,19),(54,19),(55,19),(56,19),(57,22),(58,19),(59,19),(60,24),(61,25),(62,26),(63,27),(64,28),(65,29),(66,30),(67,31),(68,32),(69,33),(70,34),(71,35),(72,36),(73,37),(74,38),(75,39),(76,40),(77,41),(78,42),(79,43),(80,44),(81,45),(82,29),(83,47);

/*Table structure for table `books` */

DROP TABLE IF EXISTS `books`;

CREATE TABLE `books` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `title` varchar(255) NOT NULL,
  `isbn` varchar(13) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `isbn` (`isbn`)
) ENGINE=InnoDB AUTO_INCREMENT=84 DEFAULT CHARSET=latin1;

/*Data for the table `books` */

insert  into `books`(`id`,`title`,`isbn`) values (1,'Bel and the Dragon ','123828863494'),(2,'Daughters of Men ','1234404543724'),(3,'The Giant on the Hill ','1236400967773'),(4,'Marsh Lights ','1233673027750'),(5,'Mr. Wodehouse and the Wild Girl ','1232423190947'),(6,'The Fairy Castle ','1237654836443'),(7,'The Girl Who Walked a Long Way ','1230211946720'),(8,'The Runaway ','1238155430735'),(9,'The Shrubbery ','1237366725549'),(10,'Tom Underground a play ','1239633328787'),(11,'Anemones of the British Coast ','1233540471995'),(12,'Ask to Embla poem-cycle ','1237417184084'),(13,'Cassandra verse drama ','1235260611012'),(14,'Chidiock Tichbourne ','1230468662299'),(15,'The City of Is ','1233136349197'),(16,'Cromwell verse drama ','1239653041219'),(17,'Debatable Land Between This World and the Next ','1235927658929'),(18,'The Fairy Melusina epic poem ','1232341278470'),(19,'The Garden of Proserpina ','1234685512892'),(20,'Gods Men and Heroes ','1233369260356'),(21,'The Great Collector ','1237871538785'),(22,'The Grecian Way of Love ','1234003421055'),(23,'The Incarcerated Sorceress ','1233804025236'),(24,'Last Tales ','1231588537286'),(25,'Last Things ','1239338429682'),(26,'Mummy Possest poem ','1239409501196'),(27,'No Place Like home ','1239416066484'),(28,'Pranks of Priapus ','1231359225882'),(29,'Ragnarök ','1230741986307'),(30,'The Shadowy Portal ','1232294350642'),(31,'Jan Swammerdam poem ','1238329678939'),(32,'St. Bartholomew\'s Eve verse drama ','1230082140880'),(33,'Tales for innocents ','1234392912372'),(34,'Tales Told in November ','1234549242464'),(35,'Bel and the Dragon ','1239374496485'),(36,'Daughters of Men ','1235349316660'),(37,'The Giant on the Hill ','1235644620578'),(38,'Marsh Lights ','1235736344898'),(39,'Mr. Wodehouse and the Wild Girl ','1232744187226'),(40,'The Fairy Castle ','1233729213076'),(41,'The Girl Who Walked a Long Way ','1237641884608'),(42,'The Runaway ','1233964452155'),(43,'The Shrubbery ','1231273626499'),(44,'Tom Underground a play ','1238441018900'),(45,'In A Future Chalet School Girl: Mystery at Heron Lake ','1231377433718'),(46,'In Althea Joins the Chalet School: The Secret of Castle Dancing ','1232395135758'),(47,'In Carola Storms the Chalet School: The Rose Patrol in the Alps ','1234185299775'),(48,'In The Chalet School Goes To It: Gipsy Jocelyn ','1234645928899'),(49,'In Gay from China at the Chalet School: Indian Holiday and Nancy Meets a Nazi ','1230275004688'),(50,'In Jo Returns to the Chalet School: Cecily Holds the Fort and Malvina Wins Through ','1230839327111'),(51,'In Joey Goes to Oberland: Audrey Wins the Trick and Dora of the Lower Fifth ','1237588408519'),(52,'In The Chalet School and the Island: The Sea Parrot ','1236495378720'),(53,'In The Chalet School in Exile: Tessa in Tyrol ','1236588981768'),(54,'In The Mystery at the Chalet School: The Leader of the Lost Cause ','1231308608691'),(55,'In The New Mistress at the Chalet School: King\'s Soldier Maid and Swords Crossed ','1230312140169'),(56,'In A Problem for the Chalet School: A Royalist Soldier-Maid and Werner of the Alps ','1230967619568'),(57,'In Three Go to the Chalet School: Lavender Laughs in Kashmir ','1230127072745'),(58,'In Tom Tackles the Chalet School: The Fugitive of the Salt Cave and The Secret House ','1234238103911'),(59,'In Two Sams at the Chalet School: Swords for the King! ','1230886230089'),(60,'In Maids of La Rochelle: Guernsey Folk Tales ','1233675376783'),(61,'Bacon Death ','1236766330719'),(62,'Breakfast First ','1236432913317'),(63,'The Culinary Dostoevski ','1234582103529'),(64,'The Egg Laid Twice ','1236148226462'),(65,'He Kissed All Night ','1237321964604'),(66,'A History of Nebraska ','1239609581078'),(67,'Hombre ','1235105625585'),(68,'It\'s the Queen of Darkness Pal ','1237435357811'),(69,'Jack The Story of a Cat ','1233766820792'),(70,'Leather Clothes and the History of Man ','1236346938182'),(71,'Love Always Beautiful ','1233800248087'),(72,'Moose ','1232083986943'),(73,'My Dog ','1236297974136'),(74,'My Trike ','1237550454699'),(75,'The Need for Legalized Abortion ','1238912644528'),(76,'The Other Side of My Hand ','1239707352212'),(77,'Pancake Pretty ','1234761413168'),(78,'Printer\'s Ink ','1230702325223'),(79,'The Quick Forest ','1236002513635'),(80,'Sam Sam Sam ','1239666823646'),(81,'The Stereo and God ','1231316672178'),(82,'UFO vs. CBS ','1239778693754'),(83,'Vietnam Victory ','1237098200581');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
