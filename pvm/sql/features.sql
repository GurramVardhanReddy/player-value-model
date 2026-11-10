-- Player features from each player's first 90 days, and targets from the next 365.
-- Runs in Spark SQL over three temp views: fixed, live (daily rows per player)
-- and players (one row per player). Day 0 is the player's first day with a stake.
-- Parameters ${OBS_DAYS} and ${PRED_DAYS} are substituted by spark_features.py.

WITH daily AS (
    -- one row per player per calendar day, fixed-odds and live combined
    SELECT UserID, DateBet,
           SUM(stake_f) AS stake_f, SUM(win_f) AS win_f, SUM(bets_f) AS bets_f,
           SUM(stake_l) AS stake_l, SUM(win_l) AS win_l, SUM(bets_l) AS bets_l
    FROM (
        SELECT UserID, DateBet, StakeF AS stake_f, WinF AS win_f, BetsF AS bets_f,
               0D AS stake_l, 0D AS win_l, 0 AS bets_l
        FROM fixed
        UNION ALL
        SELECT UserID, DateBet, 0D, 0D, 0, StakeL, WinL, BetsL
        FROM live
    ) u
    GROUP BY UserID, DateBet
),
first_day AS (
    -- day 0 = first day with a stake (settlement-only days carry winnings but no stake)
    SELECT UserID, MIN(DateBet) AS t0
    FROM daily
    WHERE stake_f + stake_l > 0
    GROUP BY UserID
),
anchored AS (
    SELECT d.*, f.t0,
           DATEDIFF(d.DateBet, f.t0) AS day,
           d.stake_f + d.stake_l AS stake,
           d.win_f + d.win_l AS win,
           d.bets_f + d.bets_l AS bets,
           CASE WHEN d.stake_f + d.stake_l > 0 THEN 1 ELSE 0 END AS active
    FROM daily d
    JOIN first_day f USING (UserID)
),
gaps AS (
    -- days between consecutive active days inside the observation window
    SELECT UserID,
           AVG(gap)         AS mean_gap,
           MAX(gap)         AS max_gap,
           STDDEV_SAMP(gap) AS sd_gap
    FROM (
        SELECT UserID,
               day - LAG(day) OVER (PARTITION BY UserID ORDER BY day) AS gap
        FROM anchored
        WHERE active = 1 AND day < ${OBS_DAYS}
    ) g
    GROUP BY UserID
),
obs AS (
    SELECT *,
           -- rolling 7-day stake ending on this day
           SUM(stake) OVER (PARTITION BY UserID ORDER BY day
                            RANGE BETWEEN 6 PRECEDING AND CURRENT ROW) AS stake_7d,
           -- running share of the window's stake, to find when half of it was placed
           SUM(stake) OVER (PARTITION BY UserID ORDER BY day
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
             / NULLIF(SUM(stake) OVER (PARTITION BY UserID), 0) AS cum_share
    FROM anchored
    WHERE day >= 0 AND day < ${OBS_DAYS}
),
obs_agg AS (
    SELECT UserID,
           MIN(t0)                                         AS t0,
           SUM(stake)                                      AS stake_90,
           SUM(stake_l)                                    AS stake_live_90,
           SUM(win)                                        AS win_90,
           SUM(bets)                                       AS bets_90,
           SUM(active)                                     AS active_days_90,
           MAX(CASE WHEN active = 1 THEN day END)          AS last_active_day,
           SUM(CASE WHEN day < 7  THEN stake ELSE 0 END)   AS stake_wk1,
           SUM(CASE WHEN day < 30 THEN stake ELSE 0 END)   AS stake_m1,
           SUM(CASE WHEN day >= 60 THEN stake ELSE 0 END)  AS stake_m3,
           SUM(CASE WHEN day >= 60 THEN active ELSE 0 END) AS active_days_m3,
           SUM(CASE WHEN day >= 76 THEN active ELSE 0 END) AS active_days_last14,
           MAX(stake)                                      AS max_day_stake,
           MAX(stake_7d)                                   AS max_7d_stake,
           MIN(CASE WHEN cum_share >= 0.5 THEN day END)    AS half_stake_day,
           SUM(CASE WHEN bets_l > 0 THEN 1 ELSE 0 END)     AS live_days_90
    FROM obs
    GROUP BY UserID
),
target AS (
    SELECT UserID,
           SUM(stake)                                      AS y_turnover,
           SUM(stake - win)                                AS y_ggr,
           SUM(active)                                     AS y_active_days,
           MAX(CASE WHEN day < ${OBS_DAYS} + 90 AND active = 1 THEN 1 ELSE 0 END) AS y_retained_90
    FROM anchored
    WHERE day >= ${OBS_DAYS} AND day < ${OBS_DAYS} + ${PRED_DAYS}
    GROUP BY UserID
)
SELECT o.*,
       g.mean_gap, g.max_gap, g.sd_gap,
       CAST(p.CountryID AS INT) AS CountryID,
       CAST(p.Gender AS INT)    AS Gender,
       CAST(p.BirthYear AS INT) AS BirthYear,   -- 'NA' in the file becomes NULL
       DATEDIFF(o.t0, p.DateReg)                           AS reg_to_first_bet,
       DATEDIFF(p.Date1Dep, p.DateReg)                     AS reg_to_first_dep,
       COALESCE(t.y_turnover, 0)                           AS y_turnover,
       COALESCE(t.y_ggr, 0)                                AS y_ggr,
       COALESCE(t.y_active_days, 0)                        AS y_active_days,
       COALESCE(t.y_retained_90, 0)                        AS y_retained_90
FROM obs_agg o
JOIN players p USING (UserID)
LEFT JOIN gaps g USING (UserID)
LEFT JOIN target t USING (UserID)
