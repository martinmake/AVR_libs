#include <wchar.h>
#include <math.h>
#include <algorithm>

#include "cabs/widgets/history_graph.h"
#include "cabs/application.h"

HistoryGraph::HistoryGraph(void)
{
}

HistoryGraph::~HistoryGraph(void)
{
}

void HistoryGraph::draw_inside(void) const
// ☭
{
	if (m_data.size() == 0)
		return;

	int w, h;
	w = getmaxx(m_win);
	h = getmaxy(m_win);

	float min_val = *std::min_element(m_data.begin(), m_data.end());
	float max_val = *std::max_element(m_data.begin(), m_data.end());

	float scale  = (h - 1) / (float) (abs(min_val) + abs(max_val));
	int   origin = (max_val < 0 ? 0 : max_val) * scale;

	wchar_t blocks[] = { L'▁', L'▂', L'▃', L'▄', L'▅', L'▆', L'▇' };
	uint8_t resolution = sizeof(blocks) / sizeof(wchar_t);

	wattron(m_win, Cabs::Colors::RED_BLACK);
	wmove(m_win, origin, 0);
	for (uint16_t i = 0; i < w - (int) m_data.size(); i++)
		wprintw(m_win, "%C", L'▁');
	for (float value : m_data)
	{
		if (value == 0)
			continue;

		bool is_negative = value < 0 ? true : false;
		value = abs(value) * scale;

		int x = getcurx(m_win);

		if (is_negative)
		{
			wprintw(m_win, "%C", L'▁');
			wmove(m_win, origin + 1, x);
		}

		for (; value >= 1; value--)
		{
			int y = getcury(m_win);
			waddch(m_win, ' ' | A_REVERSE);
			wmove(m_win, y + (is_negative ? +1 : -1), x);
		}

		if (value != 0)
		{
			if (is_negative)
			{
				wattron(m_win, A_REVERSE);
				wprintw(m_win, "%C", blocks[resolution - 1 - (int) (resolution * value)]);
				wattroff(m_win, A_REVERSE);
			}
			else
			{
				wprintw(m_win, "%C", blocks[(int) (resolution * value)]);
			}
		}
		wmove(m_win, origin, x + 1);
	}
	wattroff(m_win, Cabs::Colors::RED_BLACK);
}
