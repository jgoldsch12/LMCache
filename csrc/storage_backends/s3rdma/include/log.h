/*
 * Copyright (c) 2026 Dell Inc. or its subsidiaries.
 *
 * Dell Confidential.
 * This software contains proprietary and confidential information of Dell Inc.
 * and is subject to applicable non-disclosure agreements.
 * Unauthorized copying, modification, distribution, or use is strictly prohibited.
 */
#ifndef S3RDMA_LOG_H
#define S3RDMA_LOG_H

/**
 * @file log.h
 * @brief Lightweight structured logging for S3 RDMA client.
 *
 * Provides five log levels (TRACE, DEBUG, INFO, WARN, ERROR) with
 * automatic timestamps and thread IDs.  Usable from both C and C++.
 * All log output goes to stderr; program output (results, CSV) stays
 * on stdout.
 *
 * Format:  [YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [tid:N] message
 */

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LOG_LVL_TRACE = 0,
    LOG_LVL_DEBUG = 1,
    LOG_LVL_INFO  = 2,
    LOG_LVL_WARN  = 3,
    LOG_LVL_ERROR = 4,
} s3log_level_t;

/** Current global log level (default: LOG_LVL_INFO). */
extern s3log_level_t g_s3log_level;

void s3log_set_level(s3log_level_t level);
s3log_level_t s3log_get_level(void);

/**
 * Parse a log level name string (case-insensitive).
 * Returns LOG_LVL_INFO for unrecognised strings.
 */
s3log_level_t s3log_level_from_string(const char *str);

/** Return the canonical upper-case name for a log level. */
const char *s3log_level_to_string(s3log_level_t level);

/**
 * Write a structured log message at the given level.
 * Format: [YYYY-MM-DD HH:MM:SS.mmm] [LEVEL] [tid:N] message\n
 */
void s3log_write(s3log_level_t level, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

/**
 * Log an error with the current errno description (like perror).
 * Format: [timestamp] [ERROR] [tid:N] prefix: strerror(errno)\n
 */
void s3log_perror(const char *prefix);

/* Convenience macros -- the level check avoids function-call overhead
   when the message would be filtered out. */
#define LOG_ERROR(...) do { if (g_s3log_level <= LOG_LVL_ERROR) s3log_write(LOG_LVL_ERROR, __VA_ARGS__); } while(0)
#define LOG_WARN(...)  do { if (g_s3log_level <= LOG_LVL_WARN)  s3log_write(LOG_LVL_WARN,  __VA_ARGS__); } while(0)
#define LOG_INFO(...)  do { if (g_s3log_level <= LOG_LVL_INFO)  s3log_write(LOG_LVL_INFO,  __VA_ARGS__); } while(0)
#define LOG_DEBUG(...) do { if (g_s3log_level <= LOG_LVL_DEBUG) s3log_write(LOG_LVL_DEBUG, __VA_ARGS__); } while(0)
#define LOG_TRACE(...) do { if (g_s3log_level <= LOG_LVL_TRACE) s3log_write(LOG_LVL_TRACE, __VA_ARGS__); } while(0)

/** Replacement for perror() that goes through the structured logger. */
#define LOG_PERROR(prefix) do { if (g_s3log_level <= LOG_LVL_ERROR) s3log_perror(prefix); } while(0)

#ifdef __cplusplus
}
#endif

#endif /* S3RDMA_LOG_H */
