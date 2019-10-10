#ifndef _CABS_WIDGETS_GRAPH_H_
#define _CABS_WIDGETS_GRAPH_H_

#include <vector>

#include "cabs/widget.h"

class Graph : public Widget
{
	private:
		// std::string m_text;

	public:
		Graph(void);
		~Graph(void);

	protected:
		void draw_inside(void) const override;

	// GETTERS
	public:
		// const std::string& text   (void) const;

	// SETTERS
	public:
		// void text   (const std::string& new_text   );
};

// GETTERS
// inline const std::string& Graph::text(void) const
// {
// 	return m_text;
// }

// GETTERS
// inline void Graph::text(const std::string& new_text)
// {
// 	m_text = new_text;
// }

#endif
