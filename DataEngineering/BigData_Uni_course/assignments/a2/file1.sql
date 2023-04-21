SELECT * from users LIMIT 10;


select user_id, checkins.venue_id, category,
       utc_time + (timezone_offset_mins * interval '1 minute') as local_datetime

FROM (SELECT venue_id, category FROM pois where country = 'RU') as pois_ru
JOIN checkins
ON checkins.venue_id  = pois_ru.venue_id;

select distinct friend_id from (
    select user_id
       FROM (SELECT venue_id, category FROM pois where country = 'RU') as pois_ru
                JOIN checkins
                     ON checkins.venue_id = pois_ru.venue_id
       ) as ru_users
JOIN friendsbefore
ON friendsbefore.user_id = ru_users.user_id;


with before as(
        select users.user_id, coalesce(_before.friends_before, 0) as num_friends from
        (select user_id, count(*) as friends_before
         from friendsbefore
         group by user_id) as _before
        right join users on users.user_id = _before.user_id
)

select before.user_id, before.num_friends as num_friends_before, after.num_friends as num_friends_after, after.num_friends - before.num_friends as diff
from before
join (
        select users.user_id, coalesce(_after.friends_after, 0) as num_friends from
        (select user_id, count(*) as friends_after
         from friendsafter
         group by user_id) as _after
        right join users on users.user_id = _after.user_id
) as after
on before.user_id = after.user_id;


with top_n_friends as(
        select friend_id
        from (select user_id
                from users
                limit 1000) as users_n
                join friendsbefore on users_n.user_id = friendsbefore.user_id
)
select distinct category
from (top_n_friends
        join checkins on checkins.user_id = top_n_friends.friend_id) as friend_checkins
join (select venue_id, category from pois where country = 'RU') as ru_pois on friend_checkins.venue_id = ru_pois.venue_id;




select distinct category from
(select friend_id
    from (select user_id from users limit 1000) as top_users
           join friendsbefore on friendsbefore.user_id = top_users.user_id) as friends_top
, checkins, pois
where pois.country = 'RU' AND checkins.venue_id = pois.venue_id AND friends_top.friend_id = checkins.user_id;



with post_office_users as(
        select distinct user_id
        from (select venue_id
              from pois
              where category = 'Post Office') as post_office_venues
        join checkins
        on post_office_venues.venue_id = checkins.venue_id
)
select f2_id from (
    select distinct t2.friend_id as "f2_id" from friendsbefore as t1
    join friendsbefore as t2
        on t2.user_id = t1.friend_id
        and t1.user_id in (select user_id from post_office_users) -- making sure the initial users checked in the post office
        and t2.friend_id not in (select friend_id from friendsbefore)) -- make sure the friends are indeed 2nd degree friends
as friends_level_2

join friendsbefore as f
    on f2_id = f.user_id
    and f.friend_id not in (select friend_id from post_office_users)

where f.friend_id not in (select f2_id from friends_level_2);



with post_office_users as(
        select distinct user_id
        from (select venue_id
              from pois
              where category = 'Post Office') as post_office_venues
        join checkins
        on post_office_venues.venue_id = checkins.venue_id
)
select distinct * from
(select friendsbefore.friend_id
from (select friendsbefore.friend_id
        from
                (select friendsbefore.friend_id
                 from post_office_users
                 join friendsbefore
                 on post_office_users.user_id = friendsbefore.user_id
                ) as friends1
        join friendsbefore
        on friends1.friend_id = friendsbefore.user_id
        ) as friends2
join friendsbefore
on friends2.friend_id = friendsbefore.user_id
) as friends3;




