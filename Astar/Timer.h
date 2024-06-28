/******************************************************************************
 * timer.h
 * record the elapsed time in microseconds (10^{-6} second)
 *****************************************************************************/

#ifndef _TIMER_LJ_
#define _TIMER_LJ_
#include <chrono>
#include <cstdlib>

 /*
 #include <sys/time.h>
 class Timer {
 public:
	 Timer() { m_start = timestamp(); }
	 void restart() { m_start = timestamp(); }
	 long long elapsed() { return timestamp() - m_start; }

 private:
	 long long m_start;

	 // Returns a timestamp ('now') in microseconds
	 long long timestamp() {
		 struct timeval tp;
		 gettimeofday(&tp, nullptr);
		 return ((long long)(tp.tv_sec)) * 1000000 + tp.tv_usec;
	 }
 };
 */
class Timer {
private:
	std::chrono::system_clock::time_point m_start;

public:
	Timer() {
		m_start = std::chrono::system_clock::now();
	}

public:
	/*string to_str(void) {
		std::time_t start_t = std::chrono::system_clock::to_time_t(m_start);
		return to_string(start_t);
	}*/

	void restart() {
		m_start = std::chrono::system_clock::now();
	}

	double elapsed(void) {
		std::chrono::system_clock::time_point end;
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - m_start).count();
		return duration;
	}
};

#endif
