/*
 * Copyright (c) 2026 Dell Inc. or its subsidiaries.
 *
 * Dell Confidential.
 * This software contains proprietary and confidential information of Dell Inc.
 * and is subject to applicable non-disclosure agreements.
 * Unauthorized copying, modification, distribution, or use is strictly prohibited.
 */
/**
 * @file log.cpp
 * @brief Structured logging implementation.
 *
 * All log output is written to stderr via a single fprintf call per
 * message so that lines stay atomic even from concurrent threads.
 */

#include "log.h"

#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <sys/syscall.h>
#include <unistd.h>

s3log_level_t g_s3log_level = LOG_LVL_INFO;

static const char *level_names[] = {
    "TRACE", "DEBUG", "INFO", "WARN", "ERROR"
};

void s3log_set_level(s3log_level_t level) {
    g_s3log_level = level;
}

s3log_level_t s3log_get_level(void) {
    return g_s3log_level;
}

s3log_level_t s3log_level_from_string(const char *str) {
    if (!str) return LOG_LVL_INFO;
    if (strcasecmp(str, "trace") == 0) return LOG_LVL_TRACE;
    if (strcasecmp(str, "debug") == 0) return LOG_LVL_DEBUG;
    if (strcasecmp(str, "info")  == 0) return LOG_LVL_INFO;
    if (strcasecmp(str, "warn")  == 0) return LOG_LVL_WARN;
    if (strcasecmp(str, "warning") == 0) return LOG_LVL_WARN;
    if (strcasecmp(str, "error") == 0) return LOG_LVL_ERROR;
    return LOG_LVL_INFO;
}

const char *s3log_level_to_string(s3log_level_t level) {
    if (level >= LOG_LVL_TRACE && level <= LOG_LVL_ERROR)
        return level_names[level];
    return "UNKNOWN";
}

void s3log_write(s3log_level_t level, const char *fmt, ...) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    gmtime_r(&ts.tv_sec, &tm);

    long tid = syscall(SYS_gettid);

    /* Build the user message first, then emit header + message in one
       fprintf call so lines stay atomic on line-buffered stderr. */
    char msg[2048];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);

    fprintf(stderr,
            "[%04d-%02d-%02d %02d:%02d:%02d.%03ld] [%-5s] [tid:%ld] %s\n",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec,
            ts.tv_nsec / 1000000,
            level_names[level], tid, msg);
}

void s3log_perror(const char *prefix) {
    int saved = errno;
    s3log_write(LOG_LVL_ERROR, "%s: %s", prefix, strerror(saved));
}
