drop table if exists  pois CASCADE ;
drop table if exists  checkins CASCADE;
drop table if exists  friendsbefore CASCADE ;
drop table if exists  friendsafter;
drop table if exists  users CASCADE;

create table if not exists Users(
user_id int primary key not null
);

create table if not exists Pois(
venue_id VARCHAR(24) primary key not null,
latitude decimal,
longitude decimal,
category VARCHAR(50),
country VARCHAR(2)
);

create table if not exists Checkins(
user_id int,
venue_id VARCHAR(24),
utc_time timestamp,
timezone_offset_mins BIGINT not null
);

create table if not exists FriendsBefore(
user_id int,
friend_id int
);

create table if not exists FriendsAfter(
user_id int,
friend_id int
);

alter table FriendsBefore
add constraint fk_User_user_id_friends_before
foreign key (user_id)
references Users (user_id);

alter table FriendsBefore
add constraint fk_User_friend_id_friends_before
foreign key (friend_id)
references Users (user_id);

alter table FriendsAfter
add constraint fk_User_user_id_friends_after
foreign key (user_id)
references Users (user_id);

alter table FriendsAfter
add constraint fk_User_friend_id_friends_after
foreign key (friend_id)
references Users (user_id);

alter table Checkins
add constraint fk_User_user_id_checkins
foreign key (user_id)
references Users (user_id);

alter table Checkins
add constraint fk_Venue_venue_id_checkins
foreign key (venue_id)
references Pois (venue_id);

-- load the data to the table
\copy Users from  'data_files\my_users.tsv' delimiter E'\t' csv header;
\copy pois from  'data_files\my_POIs.tsv' delimiter E'\t' csv header;
\copy friendsbefore from  'data_files\my_friendship_before.tsv' delimiter E'\t' csv header;
\copy friendsafter from  'data_files\my_friendship_after.tsv' delimiter E'\t' csv header;
\copy checkins from  'data_files\my_checkins_anonymized.tsv' delimiter E'\t' csv header;


