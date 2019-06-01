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

	public:
		HistoryGraph(void);
		~HistoryGraph(void);

	protected:
		void draw_inside(void) const override;

	// OPERATORS
	public:
		HistoryGraph& operator<<(float value);

	// GETTERS
	public:
		const std::list<float>& data(void) const;

	// SETTERS
	public:
		void data(const std::list<float>& new_data);
};

// OPERATORS
inline HistoryGraph& HistoryGraph::operator<<(float value)
{
	m_data.push_back(value);

	if((int) m_data.size() > m_size.w())
		m_data.pop_front();

	return *this;
}

// GEfTTERS
inline const std::list<float>& HistoryGraph::data(void) const
{
	return m_data;
}

// SETTERS
inline void HistoryGraph::data(const std::list<float>& new_data)
{
	m_data = new_data;
}

#endif
