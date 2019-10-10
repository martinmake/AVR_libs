#ifndef _CABS_WIDGETS_HISTORY_GRAPH_H_
#define _CABS_WIDGETS_HISTORY_GRAPH_H_

#include <list>

#include "cabs/widget.h"

class HistoryGraph : public Widget
{
	private:
		std::list<float> m_data;
		std::string m_x_label;
		std::string m_y_label;

	private:
		bool m_is_centered = false;

	public:
		HistoryGraph(void);
		~HistoryGraph(void);

	public:
		void draw_inside(void) const override;
		void resize_inside(void) override;

	// OPERATORS
	public:
		HistoryGraph& operator<<(float value);

	// GETTERS
	public:
		const std::list<float>& data       (void) const;
		      bool              is_centered(void) const;

	// SETTERS
	public:
		void data       (const std::list<float>& new_data       );
		void is_centered(      bool              new_is_centered);
};

// OPERATORS
inline HistoryGraph& HistoryGraph::operator<<(float value)
{
	m_data.push_back(value);

	if ((int) m_data.size() > m_size.w())
		m_data.pop_front();

	return *this;
}

// GETTERS
inline const std::list<float>& HistoryGraph::data(void) const
{
	return m_data;
}
inline bool HistoryGraph::is_centered(void) const
{
	return m_is_centered;
}

// SETTERS
inline void HistoryGraph::data(const std::list<float>& new_data)
{
	m_data = new_data;
}
inline void HistoryGraph::is_centered(bool new_is_centered)
{
	m_is_centered = new_is_centered;
}

#endif
