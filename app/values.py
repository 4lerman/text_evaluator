from pydantic import BaseModel

# Languages that route to Cyrillic-script POS / description variants.
CYRILLIC_LANGS: frozenset[str] = frozenset({"ru", "kk", "uk", "bg", "mk", "sr"})


class Value(BaseModel):
    """A D.R.I.V.E. company value with descriptions in three languages."""

    code: str
    name: str
    description_en: str
    description_ru: str
    description_kk: str
    instruction: str = ""
    rubric: str = ""
    reference_answer: str = ""

    def description_for(self, lang: str) -> str:
        """Return the description in the language closest to *lang*.

        Args:
            lang: ISO 639-1 language code detected from candidate input.

        Returns:
            The matching-language description string.
        """
        if lang == "kk":
            return self.description_kk
        if lang in CYRILLIC_LANGS:
            return self.description_ru
        return self.description_en


DRIVE_VALUES: list[Value] = [
    Value(
        code="D",
        name="Disciplined Resilience / Дисциплинированная стойкость / Тәртіпті төзімділік",
        description_en=(
            "Deliberate emotional self-regulation: using concrete coping habits, recovery rituals, or structured routines "
            "to stay functional under sustained pressure. Maintaining composure through a deliberate strategy, not just "
            "passive endurance. Perseverance backed by healthy habits: regular exercise, sleep discipline, mindfulness. "
            "Determination to complete challenging work through proactive stress management."
        ),
        description_ru=(
            "Осознанная эмоциональная саморегуляция: конкретные стратегии совладания, режим восстановления "
            "и структурированные привычки для сохранения работоспособности под давлением. "
            "Самообладание как результат целенаправленных действий, а не пассивной выносливости. "
            "Настойчивость, подкреплённая здоровыми привычками: физические упражнения, режим сна, осознанность."
        ),
        description_kk=(
            "Саналы эмоционалды өзін-өзі реттеу: қысым астында жұмысқа қабілетті сақтау үшін нақты бейімделу стратегиялары, "
            "қалпына келтіру рәсімдері және құрылымдалған әдеттер. Енжар шыдамдылық емес, мақсатқа бағытталған әрекеттер "
            "арқылы сабырлылықты сақтау. Дене жаттығулары, ұйқы тәртібі, зейін сияқты салауатты әдеттермен бекітілген табандылық."
        ),
        instruction=(
            "Evaluate whether the candidate demonstrates Disciplined Resilience: "
            "emotional self-regulation, healthy habits, and determination when facing setbacks or pressure."
        ),
        rubric=(
            "score 1: No evidence of emotional regulation, discipline, or perseverance. The candidate avoids or glosses over setbacks.\n"
            "score 2: Mentions a challenge but shows no concrete self-regulation. Resilience is implied but not demonstrated.\n"
            "score 3: Describes a setback and a reasonable response. Shows some emotional awareness but lacks specificity around personal discipline or habit.\n"
            "score 4: Clearly demonstrates staying focused under pressure with specific actions taken to maintain discipline or protect the team from burnout.\n"
            "score 5: Explicitly shows emotional self-regulation, structured discipline (e.g. workflow adjustments), and determination through a significant setback with measurable team impact."
        ),
        reference_answer=(
            "When our investor pulled out, I stayed calm and kept the team on track. "
            "I maintained structured daily workflows to prevent burnout, communicated transparently, "
            "and we recalibrated without losing momentum."
        ),
    ),
    Value(
        code="R",
        name="Responsible Innovation / Ответственные инновации / Жауапты инновация",
        description_en="Creative problem-solving, ethical use of technology, hypothesis testing, data-driven decision-making.",
        description_ru="Творческое решение проблем, этичное применение технологий, проверка гипотез, принятие решений на основе данных.",
        description_kk="Шығармашылық мәселе шешу, технологияны этикалық пайдалану, гипотезаларды тексеру, деректерге негізделген шешімдер.",
        instruction=(
            "Evaluate whether the candidate demonstrates Responsible Innovation: "
            "creative problem-solving, ethical use of technology, hypothesis testing, and data-driven decision-making."
        ),
        rubric=(
            "score 1: No mention of data, ethics, or systematic problem-solving. Decisions appear arbitrary or instinct-driven.\n"
            "score 2: Mentions creativity or technology but with no ethical awareness, hypothesis testing, or data validation.\n"
            "score 3: References data or testing in some form, but the process is vague. Ethical considerations are absent or superficial.\n"
            "score 4: Demonstrates hypothesis-driven decision-making with real data and shows awareness of ethical constraints (e.g. privacy).\n"
            "score 5: Explicitly validates assumptions through live testing, cites hard data before committing resources, and demonstrates clear ethical guardrails such as data privacy or responsible deployment."
        ),
        reference_answer=(
            "We tested every assumption with live users before committing to development. "
            "I insisted on strict data privacy protocols and validated our pivot with hard data "
            "rather than instinct."
        ),
    ),
    Value(
        code="I",
        name="Insightful Vision / Проницательное видение / Болжамды көзқарас",
        description_en=(
            "Stepping back from the obvious explanation to trace root causes others missed. "
            "Mapping the full system — customer journey, product funnel, org process — to discover the real bottleneck. "
            "Questioning the assumption the team takes for granted before jumping to a solution. "
            "Finding non-obvious insight by analyzing underlying data patterns and hidden dependencies. "
            "Reframing the problem entirely before acting. Reaching counter-intuitive but evidence-backed conclusions. "
            "Seeing what everyone else overlooked by zooming out and examining the whole picture."
        ),
        description_ru=(
            "Отступить от очевидного объяснения и найти первопричины, которые упустили другие. "
            "Составить карту всей системы — клиентского пути, воронки продукта, организационного процесса — чтобы обнаружить настоящее узкое место. "
            "Поставить под сомнение предположения команды до того, как предлагать решение. "
            "Обнаружить неочевидный инсайт через анализ скрытых зависимостей и данных. "
            "Полностью переосмыслить проблему перед тем, как действовать. Прийти к нестандартному, но обоснованному выводу."
        ),
        description_kk=(
            "Анық түсіндірмеден шегініп, басқалар байқамаған түбегейлі себептерді іздеу. "
            "Клиент жолын, өнім шұңқырын немесе ұйымдастырушылық процесті толық картаға түсіріп, нақты кедергіні табу. "
            "Шешім ұсынғанға дейін команданың болжамдарын сынға алу. "
            "Жасырын тәуелділіктер мен деректерді талдау арқылы анық емес түсінік алу. "
            "Іс-әрекет жасамас бұрын мәселені түбегейлі қайта тұжырымдау. Күтпеген, бірақ дәлелді қорытындыға жету."
        ),
        instruction=(
            "Evaluate whether the candidate demonstrates Insightful Vision: "
            "systems thinking, foresight, analysis, and well-balanced judgment."
        ),
        rubric=(
            "score 1: Reactive thinking only. No evidence of analysis, foresight, or systems-level awareness.\n"
            "score 2: Identifies a problem only after it becomes obvious. No forward-looking analysis or broader context.\n"
            "score 3: Shows some analytical thinking and recognizes trends, but the insight is narrow or retrospective rather than anticipatory.\n"
            "score 4: Demonstrates forward-looking analysis with a clear connection between observed signals and strategic decisions.\n"
            "score 5: Explicitly identifies a systemic trend before others, uses that foresight to drive a proactive strategy, and connects micro-observations to macro impact."
        ),
        reference_answer=(
            "I analyzed usage trends and identified the demographic shift toward mobile before it became obvious. "
            "I proposed a strategic pivot early, anticipating where the market was heading rather than reacting to it."
        ),
    ),
    Value(
        code="V",
        name="Values-Driven Leadership / Ценностно-ориентированное лидерство / Құндылыққа негізделген көшбасшылық",
        description_en="Dignity, inclusion, dialogue, community service, leading by example, ethical leadership.",
        description_ru="Достоинство, инклюзивность, диалог, служение обществу, лидерство через пример, этичное руководство.",
        description_kk="Қадір-қасиет, инклюзивтілік, диалог, қоғамға қызмет ету, үлгі арқылы басшылық ету, этикалық көшбасшылық.",
        instruction=(
            "Evaluate whether the candidate demonstrates Values-Driven Leadership: "
            "dignity, inclusion, dialogue, and learning through service."
        ),
        rubric=(
            "score 1: No mention of others' perspectives, inclusion, or collaborative decision-making. Leadership is purely directive.\n"
            "score 2: References a team but shows no evidence of active inclusion, dialogue, or respect for individual contributions.\n"
            "score 3: Mentions team collaboration and some inclusion but lacks specificity around how diverse voices were integrated or empowered.\n"
            "score 4: Actively includes diverse team members in decision-making, fosters psychological safety, and demonstrates respect for dignity and contribution.\n"
            "score 5: Explicitly facilitates inclusive dialogue across seniority levels, ensures shared ownership, and shows how service-oriented leadership produced better collective outcomes."
        ),
        reference_answer=(
            "I ran inclusive workshops with the full team, ensuring junior and senior voices shaped our direction equally. "
            "Everyone felt ownership. Decisions were made through dialogue, not top-down mandates."
        ),
    ),
    Value(
        code="E",
        name="Entrepreneurial Execution / Предпринимательское исполнение / Кәсіпкерлік орындалым",
        description_en="Opportunity seeking, building partnerships, financial literacy, storytelling, initiative and action.",
        description_ru="Поиск возможностей, построение партнёрств, финансовая грамотность, сторителлинг, инициатива и действие.",
        description_kk="Мүмкіндіктерді іздеу, серіктестіктер орнату, қаржылық сауаттылық, оқиға айту, бастамашылық және іс-әрекет.",
        instruction=(
            "Evaluate whether the candidate demonstrates Entrepreneurial Execution: "
            "opportunity seeking, partnerships, financial literacy, and storytelling."
        ),
        rubric=(
            "score 1: No evidence of opportunity recognition, partnership building, or results-driven execution.\n"
            "score 2: Mentions an outcome but it appears accidental rather than driven by proactive opportunity-seeking or negotiation.\n"
            "score 3: Shows some initiative in finding resources or partnerships but the execution lacks specificity or measurable results.\n"
            "score 4: Proactively identifies and pursues an opportunity, executes with a clear strategy, and achieves a concrete outcome.\n"
            "score 5: Identifies a non-obvious opportunity, negotiates a strategic partnership using data-backed storytelling, and delivers a measurable outcome that proves financial and/or social viability."
        ),
        reference_answer=(
            "When funding fell through, I identified an NGO partnership opportunity, negotiated a deal "
            "using our data-backed prototype, and expanded our launch base by 40% — proving both "
            "financial and social viability."
        ),
    ),
]
