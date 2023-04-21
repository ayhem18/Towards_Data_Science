with pois_ru as(
    select venue_id, category
    from pois
    where country = 'RU'
)


select user_id, checkins.venue_id, category, utc_time + (timezone_offset_mins * interval '1 minute') as local_datetime
from pois_ru
join checkins on pois_ru.venue_id = checkins.venue_id
ORDER BY user_id;



with ru_users as(
        with ru_pois as(
            select venue_id
            from pois
            where country = 'RU'
        )
    select user_id
    from ru_pois
    join checkins on ru_pois.venue_id = checkins.venue_id
)
select distinct friend_id
from ru_users
join friendsbefore on ru_users.user_id = friendsbefore.user_id;




