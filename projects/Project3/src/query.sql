WITH first_registration AS (
    SELECT user_id,
        MIN(registration_date) AS first_registration_date
    FROM retention_users
    GROUP BY user_id
),
total_users AS (
    SELECT COUNT(DISTINCT user_id) AS total_users
    FROM first_registration
    WHERE first_registration_date < CURRENT_DATE - INTERVAL '7 days'
),
activity_by_week AS (
    SELECT EXTRACT(
            WEEK
            FROM al.date
        ) - EXTRACT(
            WEEK
            FROM fr.first_registration_date
        ) AS week_number,
        COUNT(DISTINCT al.user_id) AS active_users
    FROM retention_users_activity al
        JOIN first_registration fr ON al.user_id = fr.user_id
    WHERE EXTRACT(
            WEEK
            FROM al.date
        ) - EXTRACT(
            WEEK
            FROM fr.first_registration_date
        ) >= 0
    GROUP BY EXTRACT(
            WEEK
            FROM al.date
        ) - EXTRACT(
            WEEK
            FROM fr.first_registration_date
        )
)
SELECT week,
    active_users,
    total_users,
    CASE
        WHEN total_users = 0 THEN 0
        ELSE active_users::float / total_users
    END AS retention_percentage
FROM (
        SELECT week_number AS week,
            active_users,
            (
                SELECT total_users
                FROM total_users
            ) AS total_users
        FROM activity_by_week
    ) subquery
WHERE week != 0
ORDER BY week;