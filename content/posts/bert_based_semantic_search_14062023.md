---
title: "BERT based Semantic Search (12 June 2023)"
date: 2023-06-09T10:15:27+08:00
tags: ['blog']
---

Previously, we looked at building a semantic search engine using TF-IDF matrix as its embeddings. In this post, we will look into using BERT embeddings for our similarity search

We use a similar set of data as we did previously.

| jobId | jobUrl | jobTitle | jobDescription | datePosted | companyId | companyIdNormalised | companyName | rawWageMin | rawWageMax | sourceName | qualifications |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 00d9917e95ebbb58d237e90b5a01095947c31fbe119d82... | https://www.efinancialcareers.sg/jobs-Singapor... | Change Manager, IBOR Transition Programme | Change Manager, IBOR Transition Programme<br/>... | 2022-04-01 | 7cc4c2d8b4893e7c64265beccd30d4c1b644cf8b57a9e8... | 06634f73009b4765beafae5f98c0996b33870a3d34fa87... | Standard Chartered Bank | 0 | 0 | E-FinancialCareer |
| 1 | 0104963e8e1289488f2ff96edfe95dddc9ab84231b37a5... | https://www.efinancialcareers.sg/jobs-Singapor... | Analyst, KYC Analyst, Corporate Banking, Insti... | Analyst, KYC Analyst, Corporate Banking, Insti... | 2022-04-01 | c3475240458aa07566e1db7eec98affa5d85d8bd2b9577... | 1810faaf5f96a398f5b43df1b80809dbf5b7935f94a5f7... | DBS Bank Limited | 0 | 0 | E-FinancialCareer |
| 2 | 01561a39ff31372551e0be1caaf6a2c32150925f75af76... | https://www.jobstreet.com.sg/en/job/senior-leg... | (Senior) Legal Counsel, Autumn Venture - SC Ve... | About Standard CharteredWe are a leading inter... | 2022-04-01 | a1ad3581a81222507fa918dc2d978ed1db672c44415c3f... | 25fa191ddad0bb854bd7bbe811437b1c820271351c4ac8... | Autumn Life Pte. Ltd. | 0 | 0 | JobStreetSG |
| 3 | 0110a85844f5aa0b87060109b25567903d2188130391b2... | https://www.efinancialcareers.sg/jobs-Singapor... | Product/Data Analyst | About us<br/>Endowus is Asia's leading fee-onl... | 2022-04-01 | 4aac38458acbd96d2de7ceda69e0c3f9923c5c8b4773f5... | 3c7c5b38bc57d7f85029e41a219822866fe3dd6587027a... | Endowus | 0 | 0 | E-FinancialCareer |
| 4 | 02728a523b995791d2c81657a906c223cfdec4a9eb980b... | https://www.efinancialcareers.sg/jobs-Singapor... | Product Operations Associate (Account Opening) | About us<br/>Endowus is Asia's leading fee-onl... | 2022-04-01 | 4aac38458acbd96d2de7ceda69e0c3f9923c5c8b4773f5... | 3c7c5b38bc57d7f85029e41a219822866fe3dd6587027a... | Endowus | 0 | 0 | E-FinancialCareer |

```python
def clean_text(text):
    # remove htmltags and new lines/tags
    try:
        text = re.sub(r'<.[a-zA-Z]+.>', ' ', text)
        text = re.sub(r'&.[a-zA-Z]+.;', '', text)
        #text = re.sub(r'^[a-zA-Z.]', '', text)
        text = re.sub(r'httpS+s*', ' ', text)
        text = re.sub(r'\.', '', text)
        text = re.sub(r'\(', '',text)
        text = re.sub(r'\)', '',text)
        text = re.sub(r' +', ' ',text)
        text = text.lower()
    except Exception as e:
        print(f"Error: {text}")
        return text

    return text
```

```python
if raw_data.isnull().values.any():
    raw_data.dropna(how='any', inplace=True)
raw_data.reset_index(drop=True, inplace=True)
raw_data.set_axis(range(len(raw_data)), inplace=True)
```

We also do similar data cleaning and removing of rows with `NaN` values as well

```python
from sentence_transformers import SentenceTransformer, util
import torch

# we use siamese-BERT for our embedding as it maps paragraphs and sentences into a fixed vector space
model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
device = torch.device("cuda")
model.to(device)
```

The model we use is `paraphrase-distilroberta-base-v2` with [its model card here](https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v2) and [paper here](https://arxiv.org/abs/1908.10084). The siamese sentence-BERT performs better than vanilla BERT in amount of data processed and time, as it maps paragraphs or sentences into a fixed dimension of vector space.

```python
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False})
)
```

## Side Track..

What data to feed to Transformer models for embeddings?

[Source](https://stackoverflow.com/questions/63979544/using-trained-bert-model-and-data-preprocessing), [Additional Source](https://stackoverflow.com/questions/69428811/how-does-bert-word-embedding-preprocess-work)

Does preprocessing of data changes how attention module in BERT I think preprocessing will not change your output predictions. I will try to explain for each case you mentioned -

1. **stemming or lemmatization** :
Bert uses BPE (**Byte- Pair Encoding** to shrink its vocab size), so words like run and running will ultimately be decoded to **run + ##ing.**
So it's better not to convert *running* into *run* because, in some NLP problems, you need that information.
2. **De-Capitalization** - Bert provides two models
(lowercase and uncased). One converts your sentence into lowercase, and
others will not change related to the capitalization of your sentence.
So you don't have to do any changes here just select the model for your
use case.
3. **Removing high-frequency words** -
Bert uses the Transformer model, which works on the attention principal.
So when you finetune it on any problem, it will look only on those words which will impact the output and not on words which are common in all
data.

```python
%time embeddings = model.encode(raw_data['jobdata'])`
CPU times: user 8min 55s, sys: 2.28 s, total: 8min 57s
Wall time: 7min
cuda_embeddings = torch.from_numpy(embeddings).float().to(device)
```

```python
# Sample queries
queries = [
    'data scientist job in financial sector', 
    'Remote only job', 
    'Web frontend react',
    'sales executive',
    'Devops AWS',
    'Cooking job'
]
```

```python
# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(raw_data['jobdata']))

for query in queries:
    query_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, cuda_embeddings)[0]

    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print(top_results[0])
    print("\nTop 5 most similar sentences in corpus:\n")

    for score, idx in zip(top_results[0], top_results[1]):
        print(f"idx: {idx}" + " (Score: {:.4f})".format(score))
        print(f"Text:\n {raw_data['jobdata'][int(idx)]}\n")
```

## Results

```python
======================

Query: data scientist job in financial sector
tensor([0.6497, 0.6418, 0.6390, 0.6387, 0.6368], device='cuda:0')

Top 5 most similar sentences in corpus:

idx: 3138 (Score: 0.6497)
Text:
 singapore exchange limited, financial data engineer, job description amp requirementsa financial data engineer is responsible for developing, testing and maintaining architectures to improve data reliability and processing efficienciesyou will produce regular reports using financial data for leadership review, and work collaboratively across the entire finance team to analyse sgxs business performance and strategyyou will work closely with financial data analyst in forecasting future revenue and expenditures to help determine budgets for upcoming projectskey duties include: improving current methods used in the collection of data from other parts of the company while retaining the essential integrity and security of the information collected analyze financial data and create financial models for decision support identify and drive process improvements, including the creation of standard and ad-hoc reports, tools and dashboards increase productivity by developing automated reporting and forecasting tools perform market research, data mining, business intelligence and company analysis encourage the transfer of relevant knowledge to the parties that require it the mostrequirementsa good degree in one of the following subjects: analytics/ computer modelling/ science or mathematics finance/ accountancy/ economicsa finance data engineer is also expected to demonstrate the following qualities: good critical thinking skills and problem solving skills strong quantitative and analytical competency high attention to detail and the ability to perform forward-thinking forecasts based upon financial trends research proficiency in python, r, tableau good verbal and written communication skills abreast of industry updatesother qualifications and skills include: corporate finance, developing standards, quality management, problem solving, process improvement, cost accounting, statistical analysis, financial forecasting, financial planning and strategy, reporting research results, requirements analysis, financial skills-

idx: 14133 (Score: 0.6418)
Text:
 united overseas bank limited, avp, data analytics, corporate real estate services / human resource, - roles responsibilities : job responsibilities 	responsible for the communication of action plans and following-up on action items 	analyse data and perform research for market insight eg comparisons of companies based on available data on platforms such as sgx webpage or in the media 	proactively collaborate with stakeholders on data-related tasks job requirements 	degree in it or data-related fields 	at least 5 years of relevant experience in data analytics or analyst roles 	outstanding communication and interpersonal abilities 	excellent knowledge of ms office, databases and information systems 	good understanding of research methods and data analysis techniques

idx: 20592 (Score: 0.6390)
Text:
 citibank na, 22436757 data scientist, - roles responsibilities : job background/context: we are looking for a data scientist with deep analytics and statistics background who can contribute on finding new insights on data ability to conduct high-level market and business research to identify trends and opportunities the candidate will be working closely with data engineer, application developers and tts business units to realize data and ai based solutions for high impact business problems/projects through in-depth data discovery and exploration key responsibilities: identify valuable data sources and automate collection processes undertake preprocessing of structured and unstructured data analyze large amounts of information to discover trends and patterns build predictive models and machine-learning algorithms combine models through ensemble modeling present information using data visualization techniques propose solutions and strategies to business challenges collaborate with engineering and product development teams the candidate needs to be able to present back the findings to the business in a business friendly manner knowledge/experience: essential total 8 years of overall experience with minimum 5 years of experience in dealing with data analytics/modeling/mining/engineering strong problem-solving skills strong mathematical and analytical skills experience in data mining techniques knowledge of advanced statistical methods and concepts extensive knowledge of predictive modeling algorithms and frameworks experience using statistical computer languages like r and python to manipulate data and draw insights from large data sets experience working with and creating data architectures knowledge of a variety of machine learning techniques like but not limited to clustering, decision tree learning, artificial neural networks, and their real-world advantages/drawbacks desirable have been part of an agile development team in a reciprocal environment openness to learn new and emerging technologies skills: technical skills essential deep understanding of architecture and system integration experience analyzing data from third-party providers like adwords, facebook insights, google analytics, and hexagon real-time and dynamic data analysis expertise such as airline in-flight data, on-track engine data is a plus experience in both nosql databases and relational databases for example, couch, mongodb, and neo4j experience developing automated workflows python or r experience using web services like digital ocean, redshift, spark, and s3 experience visualizing and presenting data using business objects, periscope, ggplot, and d3 experience with distributed data/computing tools like map/reduce, hadoop, hive, spark, and gurobi experience with data visualization tools like tableau desirable experience working in a cloud environment with large data sets qualifications: degree/certifications essential masters degree or phd in computer science, math, engineering, or a related quantitative field demonstrated leadership and project/program management skills certification from a recognized institution on one of data analytics, machine learning and deep learning consistently demonstrates clear and concise written and verbal communication competencies soft skills you are looking for in a candidate great team player

idx: 14153 (Score: 0.6387)
Text:
 united overseas bank limited, vp, data analytics, corporate real estate services / human resource, - roles responsibilities : job responsibilities 	responsible for the communication of action plans and following-up on action items 	analyse data and perform research for market insight eg comparisons of companies based on available data on platforms such as sgx webpage or in the media 	proactively collaborate with stakeholders on data-related tasks job requirements 	degree in it or data-related fields 	at least 7 years of relevant experience in data analytics or analyst roles 	outstanding communication and interpersonal abilities 	excellent knowledge of ms office, databases and information systems 	good understanding of research methods and data analysis techniques

idx: 14135 (Score: 0.6368)
Text:
 united overseas bank limited, senior officer, data analytics, corporate real estate services / human resource, - roles responsibilities : job responsibilities 	responsible for the communication of action plans and following-up on action items 	analyse data and perform research for market insight eg comparisons of companies based on available data on platforms such as sgx webpage or in the media 	proactively collaborate with stakeholders on data-related tasks job requirements 	degree in it or data-related fields 	at least 2 years of relevant experience in data analytics or analyst roles 	outstanding communication and interpersonal abilities 	excellent knowledge of ms office, databases and information systems 	good understanding of research methods and data analysis techniques

======================

Query: Remote only job
tensor([0.4498, 0.4436, 0.4430, 0.4214, 0.4163], device='cuda:0')

Top 5 most similar sentences in corpus:

idx: 32770 (Score: 0.4498)
Text:
 credit culture pte ltd, remote work from home flexible hour, job descriptions handle customer service hotline and provide first level of support arrange daily schedule for field mechanic attend breakdown /servicing resolve customer queries efficiently and escalate problems/issues promptly and prepare quotation where necessary follow-up on outstanding cases and perform a call back to client if necessary collate and update machine data according to specified templates manage warranty submission for new machinery telehandler/mewp/generator, etc administrative duties where applicable to support operations any ad hoc duties as assigned telegram: @primpei 'for more details -

idx: 1888 (Score: 0.4436)
Text:
 universal far east pte ltd, technical specialist/technician, job descriptionprovide system installation and service at customers' site and/or in-housebasic troubleshooting, maintenance and repair on designated equipmentperform preventive maintenance and field modificationsmaintaining customer service logs and internal service records in a timely mannerto perform a routine task using prescribed proceduresmaintaining tools and test equipment and ensure they are properly calibratedother ad-hoc duties as assigned by supervisor/managerjob requirements:nitec or equivalent in electrical/electronics/mechanical/mechatronic with at least 2 years of relevant experienceself-motivated and able to work independently with minimal supervisionwilling to travel to customers' sites as and when required-

idx: 19986 (Score: 0.4430)
Text:
 universal far east pte ltd, technical specialist, - roles responsibilities : job description provide system installation and service at customers' site and/or in-house basic troubleshooting, maintenance and repair on designated equipment perform preventive maintenance and field modifications maintaining customer service logs and internal service records in a timely manner to perform a routine task using prescribed procedures maintaining tools and test equipment and ensure they are properly calibrated other ad-hoc duties as assigned by supervisor/manager job requirements: nitec or equivalent in electrical/electronics/mechanical/mechatronic with at least 2 years of relevant experience self-motivated and able to work independently with minimal supervision willing to travel to customers' sites as and when required

idx: 29740 (Score: 0.4214)
Text:
 veltiston group, appointment setting officer #urgent, roles responsibilities you are expected to be a determined and hardworking individual **and you must have a preference to work from home job tasks include: communicating with new prospects and arrange sales appointments follow up with existing clients to ensure they are happy with our service coordination work between sales and administration service you are expected to work from home, so you must be disciplined and independent

idx: 18674 (Score: 0.4163)
Text:
 nomura singapore limited, analyst - desktop support engineer - information technology, job overview the desktop support engineers role is to support and maintain organizational computer systems, desktops, workstations, vdi, laptops, peripherals and software systems that includes installing, diagnosing, repairing, maintaining, and upgrading all organizational hardware and equipment while ensuring optimal workstation performance the person will also troubleshoot problem areas in person, by telephone, or via ticketing system in a timely and accurate fashion, and provide end-user assistance where required physically in office or remotely responsibilities the candidate will join the workspace services desktop team to provide customer support across end user computing technology platform physically in office and remotely to end users perform both on-site and remote analysis, diagnosis, and resolution of complex desktop problems for end users, and recommend and implement corrective solutions, including off-site repair for remote users as needed install, configure, test, maintain, monitor, and troubleshoot end user and network hardware, peripheral devices, printing/scanning devices, presentation equipment, software, and other products in order to deliver required desktop service levels construct, install, and test customized configurations based on various platforms and operating systems collaborate and next level escalation with technology team members to ensure efficient operation of the organization's desktop computing environment receive and respond to incoming calls, emails, tickets and/or work orders regarding desktop problems inventory record keeping, monitoring and coordination and status report for relocation coordinate office restack and it equipment relocation related to end user devices ensure desktop connections from top down by performing various checks on regular basis prepare tests and applications for monitoring desktop performance, then provide performance statistics and reports liaise with third-party support and pc equipment vendors perform related duties consistent with the scope and intent of the position requirements hands-on experience with microsoft windows operating systems and microsoft office environment excellent knowledge of network security practices and anti-virus programs excellent knowledge of pc and desktop hardware excellent knowledge of pc internal components hands-on hardware troubleshooting experience extensive equipment support experience working technical knowledge of current protocols, operating systems, and standards ability to operate tools, components, and peripheral accessories able to read and understand technical manuals, procedural documentation and oem guides strong customer service orientation strong sense of work ethics and discipline in complying with regulations and policies proven problem-solving abilities and multitasking skills ability to effectively prioritize and execute tasks in a high-pressure environment good written, oral, and interpersonal communication skills in english ability to conduct research into pc and software issues and products as required ability to present ideas in business-friendly and user-friendly language ability to perform remote troubleshooting and provide clear instructions highly self-motivated and result-driven to be able to deliver independently keen attention to detail team-oriented and skilled in working within a collaborative environment lifting and transporting of moderately heavy objects, such as computers and peripherals occasionally requirement to perform work at external location eg user place or event venue preferred minimum 3 years of desktop support experience in microsoft windows operating systems and microsoft office environment solid experience in supporting financial institution and have good understating of support system/ticketing system knowledge of market data and financial operating models will be advantage diversity statement nomura is committed to an employment policy of equal opportunities, and is fundamentally opposed to any less favourable treatment accorded to existing or potential members of staff on the grounds of race, creed, colour, nationality, disability, marital status, pregnancy, gender or sexual orientation ts id: 1215920 -

======================

Query: Web frontend react
tensor([0.4562, 0.4524, 0.4420, 0.4308, 0.4244], device='cuda:0')

Top 5 most similar sentences in corpus:

idx: 34898 (Score: 0.4562)
Text:
 malayan banking berhad, frontend developer, digital & crm, responsibilities gather, document and analyze business requirements in order to help define scope of software development initiatives may include web page mock-ups and interaction prototyping produce and maintain documentation related to application software eq scope requirements, logical and technical designs, testing and implementation plans troubleshoot and implement bug fixes related to client and user reported issues architect and implement web ui based on provided wireframes and business requirements play proactive support role and take ownership of technical issues, and work with internal/ cross functional/ external team to resolve more advanced issues when necessary experience degree in computer science, computer programming, or related degree with at least 2 years relevant experience experience in es6 and frontend javascript frameworks ******** and/or react native is a must solid understanding of redux and best practices for highly performant js code knowledge of webpack and nodejs would be a huge advantage working experience of git version control proficient in bootstrap or equivalent css frameworks knowledge on sass/less css preprocessors are preferred detail-oriented with eyes sensitive to the aesthetics of ui layout -

idx: 18155 (Score: 0.4524)
Text:
 wechain fintech pte ltd, frontend engineer, responsibilities responsible for the company's high-performance, global website development, and maintenance responsible for solving various intractable diseases of the extreme front-end experience responsible for researching the development of browser plug-ins and other tools responsible for the research on the multi-end horizontal expansion of the platform requirements more than 5 years of front-end development experience, proficient in front-end technologies such as html/css/javascript, and proficient in es6 syntax at least proficient in using one of the mvvm frameworks *************** is the best, proficient in using the best practices of various technology stacks, react technology stack is preferred, and electron experience is a plus proficient in front-end performance optimization deep understanding of tcp/ip, http, websocket and other network protocols deeply understand some security aspects of front-end and even network applications possess strong architectural capabilities, can independently complete complex front-end module design and system architecture, have a deep understanding and rich experience in user experience and interactive operation procedures, and rich development experience love the front-end, understand various front-end trends and the latest technologies and solutions clear thinking, good communication skills, and teamwork skills, good at learning, summarizing, and willing to share benefits up to 20 days annual leave festive gifts flexible working hours overtime meal and transport benefits employee growth funding group insurance regular employee bonding events attractive annual variable bonus and quarterly performance incentives culture we foster our core values through active listening, caring, and supporting continuous improvement for all our staff members experience the energetic working environment with multicultural staff members with varied skills and experiences -

idx: 10883 (Score: 0.4420)
Text:
 wechain fintech pte ltd, frontend software engineer, - roles responsibilities : responsibilities responsible for the company's high-performance, global website development, and maintenance; responsible for solving various intractable diseases of the extreme front-end experience; responsible for researching the development of browser plug-ins and other tools; responsible for the research on the multi-end horizontal expansion of the platform; requirements more than 5 years of front-end development experience, proficient in front-end technologies such as html/css/javascript, and proficient in es6 syntax; at least proficient in using one of the mvvm frameworks vuejs/reactjs is the best, proficient in using the best practices of various technology stacks, react technology stack is preferred, and electron experience is a plus; proficient in front-end performance optimization; deep understanding of tcp/ip, http, websocket and other network protocols; deeply understand some security aspects of front-end and even network applications; possess strong architectural capabilities, can independently complete complex front-end module design and system architecture, have a deep understanding and rich experience in user experience and interactive operation procedures, and rich development experience; love the front-end, understand various front-end trends and the latest technologies and solutions; clear thinking, good communication skills, and teamwork skills, good at learning, summarizing, and willing to share benefits up to 20 days annual leave festive gifts flexible working hours overtime meal and transport benefits employee growth funding group insurance regular employee bonding events attractive annual variable bonus and quarterly performance incentives culture we foster our core values through active listening, caring, and supporting continuous improvement for all our staff members experience the energetic working environment with multicultural staff members with varied skills and experiences

idx: 15774 (Score: 0.4308)
Text:
 wechain fintech pte ltd, frontend engineer, - roles responsibilities : responsibilities responsible for the company's high-performance, global website development, and maintenance; responsible for solving various intractable diseases of the extreme front-end experience; responsible for researching the development of browser plug-ins and other tools; responsible for the research on the multi-end horizontal expansion of the platform; requirements more than 5 years of front-end development experience, proficient in front-end technologies such as html/css/javascript, and proficient in es6 syntax; at least proficient in using one of the mvvm frameworks vuejs/reactjs is the best, proficient in using the best practices of various technology stacks, react technology stack is preferred, and electron experience is a plus; proficient in front-end performance optimization; deep understanding of tcp/ip, http, websocket and other network protocols; deeply understand some security aspects of front-end and even network applications; possess strong architectural capabilities, can independently complete complex front-end module design and system architecture, have a deep understanding and rich experience in user experience and interactive operation procedures, and rich development experience; love the front-end, understand various front-end trends and the latest technologies and solutions; clear thinking, good communication skills, and teamwork skills, good at learning, summarizing, and willing to share benefits up to 20 days annual leave festive gifts flexible working hours overtime meal and transport benefits employee growth funding group insurance regular employee bonding events attractive annual variable bonus and quarterly performance incentives culture we foster our core values through active listening, caring, and supporting continuous improvement for all our staff members experience the energetic working environment with multicultural staff members with varied skills and experiences

idx: 40913 (Score: 0.4244)
Text:
 xendit, senior front end engineer - reporting, image/svg+xml job application -

======================

Query: sales executive
tensor([0.5740, 0.5725, 0.5714, 0.5693, 0.5688], device='cuda:0')

Top 5 most similar sentences in corpus:

idx: 13016 (Score: 0.5740)
Text:
 avis capital pte ltd, sales executives, - roles responsibilities : we are looking for professional full time direct sales executives and manager for an established consultancy job scope includes: 1 cold calling marketing leads 2 door to door sales 3 client relationship management on job training will be provided attractive renumeration package with sales target incentives and 5 day work week

idx: 10337 (Score: 0.5725)
Text:
 euler hermes singapore services pte ltd, sales manager, apac, to be advice

idx: 7715 (Score: 0.5714)
Text:
 selbyjenningssingapore, asia pacific sales & marketing director specialty chemicals, responsibilities: define the approach for the overall sale of new business and account retention develop a comprehensive sales and distribution strategy to maximise sales opportunities direct the sales team in generating proposals that define a clear path to client satisfaction and revenue growth build maintain relationships with clients and stakeholders leverage innovative ideas to maximise sales opportunities keep up to date with market trends leverage them for business opportunities requirements: minimally 8 years of b2b commercial experience in a customer facing sales role with client account accountability proven ability to influence internal and external stakeholders good leadership and management skills strong communication and analytical skills proficient in english, both written oral based in singapore

idx: 14745 (Score: 0.5693)
Text:
 ariston services pte ltd, senior sales manager - it product sales, - roles responsibilities : responsibilities: responsible to drive sales for it product of a large listed us mnc company experience of seeling it products in the asean market create regional sales plans and quotas in alignment with business objectives report on regional sales results forecast quarterly and annual profits prepare and review the annual budget for the area of responsibility analyse regional market trends and discover new opportunities for growth address potential problems and suggest prompt solutions skills required: proven work experience as a regional sales manager or similar senior sales role 7+ years it sales experience should have experience in product selling 2+ years in leadership sales position track record of superior performance metrics excellent negotiation skills strong decision-making abilities experience in ecommerce would be added advantage ability to measure and analyse key performance indicators roi and kpis ability to lead and motivate a high-performance sales team excellent communication skills strong organizational skills with a problem-solving attitude availability to travel as needed degree in sales, business administration or relevant field

idx: 16958 (Score: 0.5688)
Text:
 veltiston group, sales representative, you will be expected to carry out the strategic operations tasks of a globally recognised mnc tasks includes not only operational duties but also the gathering of information and sales we are looking for someone who is able to work under challenging conditions and deadlines experience with financial and marketing positions preferred but not a must singaporeans and prs only -

======================

Query: Devops AWS
tensor([0.6267, 0.5644, 0.5642, 0.5629, 0.5343], device='cuda:0')

Top 5 most similar sentences in corpus:

idx: 24296 (Score: 0.6267)
Text:
 ihs markit, devops engineer, we offer you an excellent opportunity to: develop and maintains mission-critical financial services systems handle deployment, automation, management and maintenance of aws cloud-based production system define and deploy systems for metrics, logging, and monitoring on aws platform design, maintain and manage tools for automation of different operational processes automate security controls, governance processes, and compliance validation on aws participate in architecture and software development activities provide direct and responsive support for urgent incidents informally train and share aws knowledge within the team integrate with, and works as part of the company's technology team work with advanced tools and methodologies be part of a top team, grow rapidly and expand areas of expertise in the fin-tech world about you if you have those: at least 3 years of industry experience with devops or similar role experience working on cloud providers aws/azure/gcp and managing infrastructure as code on them preferably using aws cloudformation and terraform experience with container orchestration services, especially kubernetes/openshift, docker experience with deploying and administering kubernetes/openshift/aws eks clusters experience with infrastructure scripting solutions using python/go, chef with or without aws opsworks, ansible etc experience administering and deploying development ci/cd tools such as git, gitlab, or jenkins familiarity with elastic search and managing relational databases on cloud familiarity around monitoring tools like new-relic or other open source alternatives excellent oral and written communication skills with a keen sense of customer service excellent problem-solving and troubleshooting skills process-oriented with great documentation skills knowledge of best practices and it operations in an always-up, always-available service what we offer the salary levels paid in the company are competitive and attractive in relation to the market there is a generous package of conditions that includes benefits such as: financing a gym, comprehensive health insurance, participation in the company's options program traded on the stock exchange and more we are flexible when it comes to combining work from home and office and in any case, we believe in a healthy balance between the two the work environment in the beautifully designed offices in israel combined with the charming staff, make our company a wonderful place to belong to -

idx: 851 (Score: 0.5644)
Text:
 dbs, sa/assoc, infrastructure cloud devops engineer, tech services, technology & operations, !*!roles and responsibilty - candidate would be part of public cloud sre team and would be responsible for managing/provisioning aws accounts/gcp projects - develop solutions for multi cloud platform so that we have capability for portability - create and manage ami pipeline for gcp/aws and other os that dbs bank will onboard for public cloud - integrate public cloud with different systems in the bank like incident management, privilege id management etc - ensure security compliace of all the resources hosted by the team - ensure uptime of all the common services like proxy/ntp/bastion etc - develop/implement custom solutions as per the requirements reporting/compliance/bau automation - provide adhoc support to application teams as needed desired skills\: - relevant certificates for aws/gcp - experienced with ci/cd pipeline and setup like jenkins/bitbucket - experienced with configuration management tool like ansible/chef/puppet etc - good hands on experience with boto3/python - expert level knowledge for atleast one cloud provider \: aws/gcp

idx: 940 (Score: 0.5642)
Text:
 dbs, infrastructure cloud devops engineer, !*!roles and responsibilty - candidate would be part of public cloud sre team and would be responsible for managing/provisioning aws accounts/gcp projects - develop solutions for multi cloud platform so that we have capability for portability - create and manage ami pipeline for gcp/aws and other os that dbs bank will onboard for public cloud - integrate public cloud with different systems in the bank like incident management, privilege id management etc - ensure security compliace of all the resources hosted by the team - ensure uptime of all the common services like proxy/ntp/bastion etc - develop/implement custom solutions as per the requirements reporting/compliance/bau automation - provide adhoc support to application teams as needed desired skills\: - relevant certificates for aws/gcp - experienced with ci/cd pipeline and setup like jenkins/bitbucket - experienced with configuration management tool like ansible/chef/puppet etc - good hands on experience with boto3/python - expert level knowledge for atleast one cloud provider \: aws/gcp

idx: 32938 (Score: 0.5629)
Text:
 iapps pte ltd, devops engineer aws certified, job descriptions: as part of the infra/devops team, the devops engineer holds the responsibility and accountability for it cloud-based applications infrastructure, network and security for government agency and large mnc projects, as well as in-house projects he/she should work closely with clients, product team ba/designers/pm, quality assurance qa team and operations to deliver projects in time with quality and reusable and scalable infrastructure and security framework and documentation devops engineer should be able to: administer, and maintain aws environments within hosted infrastructure and public clouds architect the solution for the business needs and assist with its implementation responsible for vulnerability assessment, penetration testing and recommend patches in our environment which is aws and lemp/lamp solid experience as a devops engineer in a 24x7 uptime amazon aws environment, including automation experience with configuration management tools experience with devops and infrastructure as code: aws environment and application automation utilizing cloudformation and third-party tools ci/cd pipeline setup utilising either aws services such as codebuild/commit/deploy or third-party tools such as codeship engineer with experience in aws, docker/kubernetes, ci/cd and capabilities in linux operating systems work closely with the development team and align security compliance around integration solutions responsible and participates in release management, infrastructure and system incident response develop, manage and test disaster recovery plans annually ensure application storage, archive, backup and restore procedures are functioning correctly perform system preventive such as security/os patchesmaintenance and execute annual audit and dr exercises design, develop unix scripts andexecute established plans for disaster recovery based on industrial best practices develops and updates system operation documentation/sop related to new and existing systems creates and builds physical and virtual os and optimize aws hosting cost through automated resource scaling provides customer support by seeking to understand and address the customer's needs and expectations through effective communication, and collaboration participates in on-call coverage during business and after hours to support all infrastructure and systems related incidents performs miscellaneous job related duties as requested handle multiple projects simultaneously job requirements: candidate should have at least 2 years aws experience with using a broad range of aws technologies eg ec2, rds, elb, ebd, s3, vpc, glacier, iam, cloudwatch, kms to develop and maintain an amazon aws based cloud solution, with an emphasis on best practice cloud security candidate with minimum 3 years ofrelevantworking experience in devops for cloud-based solutions candidate must possess aws certification candidates must possess at least a diploma / degree in computer science/information technology or equivalent require experience and proficiency with linux system administration background aws foundational services, such as ec2, sns, s3, cloudwatch, elb, route 53 ci/cd pipeline tools scripting experience with python, go, nodejs experience with system monitoring tools eg nagios experience with docker kubernetes experience in micro service, domain driven design and aws setup and provisioning experience with at least one rdbms eg mysql, postgresql, etc and nosql database eg mongodb, amazon documentdb etc the following are added advantages: experience with government agency projects experience with government, internal and external audits experience with mqtt, coap and other iot standards and protocols a t least 1 year experience in kubernetes team player with good communication and interpersonal skills added advantage for those who can speak both chinese in order to work with chinese speaking product and design team to understand end user requirements, use cases and translate that into a programmatic and effective technical solutions and english languages candidate has positive mindset, good time management, and is responsible, detailed and self-motivated

idx: 15538 (Score: 0.5343)
Text:
 fwd singapore pte ltd, cloud ops engineer 2-year contract, - roles responsibilities : purpose we are in the path to modernize and consolidate it infrastructure, automate workloads, and pursue next-generation innovation to continue this transformation, were seeking an experienced amazon web services aws cloud engineer with expertise in the strategy, design, development, and implementation of large-scale projects in the cloud objectives of this role work closely with engineering team to identify and implement the most optimal cloud-based solutions for the company define and document best practices and strategies regarding application deployment and infrastructure maintenance ensure application performance, uptime, and scale, maintaining high standards of code quality and thoughtful design manage cloud environments in accordance with company security guidelines explore new tech and services to help the organization in modernize responsibilities develop and implement technical efforts to ensure the highest level of uptime and quality of service qos for cloud environments through operational excellence participate in all aspects of the development life cycle for aws solutions, including planning, requirements, development, testing, and automating troubleshoot incidents, identify root cause, fix and document problems, and implement preventive measures automate common, repeatable tasks at large scale educate teams on the implementation of new cloud-based initiatives, providing associated training as required employ exceptional problem-solving skills, with the ability to see and solve issues before they affect business productivity required skills and qualifications bachelors degree in computer science, information technology, or mathematics 3+ years of experience with aws platforms good understanding of and experienced with the five pillars of a well-architected frameworks proven ability to collaborative with multi-disciplinary teams of business analysts, developers, data scientists, and subject matter experts preferred qualifications aws certifications are a plus knowledge of web services, api, rest familiar with devops methodologies and ci/cd tools such as jenkins etc well versed with linux and windows operating system for day 2 activities

======================

Query: Cooking job
tensor([0.5075, 0.5034, 0.4968, 0.4917, 0.4777], device='cuda:0')

Top 5 most similar sentences in corpus:

idx: 25799 (Score: 0.5075)
Text:
 iki concepts pte ltd, part-time cook, - roles responsibilities : description ?ô∏è minimum 3 days include one saturday or sunday ?ô∏è from 1030h to 2130h or 2200h ‚òï break time from 1500h to 1700h ? monday to friday $12/hr ? saturday to sunday $15/hr ? public holiday $18/hr job summary: the part-time cook shall prep all food ingredients, be able to cook all dishes on the menu, and ensure the food prepared is consistent and of superior quality job duties: 1 support the sous chef in ensuring the smooth running of kitchen operations 2 carry out preparation of dishes and ingredients for the shift 3 prepare and execute menu dishes to the highest standards 4 ensure food prepared is consistent and of superior quality 5 carry out daily general kitchen duties which include, but are not limited to, a setting up and cleaning kitchen equipment and workstations b cleaning of kitchen surfaces, floors, and storage areas c trash disposal 6 organise and ensure kitchen space is clean and tidy in accordance with food safety and hygiene standards 7 assisting other cooks in preparing food or helping other team members when needed 8 any other job-related duties requested from senior staff requirements: - has some kitchen experience in restaurants - preferably with relevant experience - ability to skillfully multitask - attention to detail - works well as part of a team and on individual tasks benefits: ? monthly payout with cpf contributions ? meal provided ‚úîô∏è able to convert to a full-time permanent position

idx: 19807 (Score: 0.5034)
Text:
 iki concepts pte ltd, charcoal grill cook, job summary: the commis shall prep all food ingredients, be able to cook all dishes on the menu, and ensure the food prepared is consistent and of superior quality job responsibility: support the sous chef in ensuring the smooth running of kitchen operations carry out preparation of dishes and ingredients for the shift prepare and execute menu dishes to the highest standards ensure food prepared is consistent and of superior quality handle all kitchen equipment properly and ensure they are kept in good working condition carry out daily general kitchen duties which include, but are not limited to, - setting up and cleaning of kitchen equipment and workstations - cleaning of kitchen surfaces, floors, and storage areas - trash disposal organize and ensure kitchen space is clean and tidy in accordance with food safety and hygiene standards assisting other cooks in preparing food or helping other team members when needed any other job-related duties requested from senior staff requirements: has some kitchen experience a mindset of someone being willing to learn and continuously improve ability to skilfully multitask attention to detail works well as part of a team and on individual tasks must be physically fit to lift up a heavy load of charcoal around 10 to 15 kg remuneration: starting basic salary from s$2000 per month onwards commensurate with experience overtime claimable working hours: 6 days work week fixed every monday off from 1400h to 2230h if there is a sake wine tasting event on weekends, start work at 11 am break time 1 hour restaurants are located conveniently: forum the shopping mall orchard station benefits: leaves such as annual leave, outpatient sick leave, hospitalisation leave marriage leave, childcare leave, compassionate leave, and more staff meal and uniform provided outpatient and dental reimbursement hospitalization and surgery insurance coverage internal and external training opportunities opportunity for career advancement tips sharing, yearly increment, variable bonus performance based, and other incentive programs staff discount and service award -

idx: 7912 (Score: 0.4968)
Text:
 cys global remit pte ltd, kitchen supervisor, - roles responsibilities : manage kitchen, food preparation, supplies ordering and kitchen staff for general food service and functions ensure kitchen operations are carried smoothly and effectively check quality of food and ensure that standards are met instruct cooks in the preparation, cooking, garnishing, and presentation of food order food and other supplies needed to ensure efficient operation check the quantity and quality of received products responsible for inventory management to reduce wastage and maximize profitability ensure kitchen cleanliness, food hygiene and kitchen safety practice ensure food handling, sanitation practices and hygiene at all times supervise and coordinate activities of kitchen staff kitchen administration duties perform other reasonable job duties as requested by supervisors job requirements: hands-on experience with planning menus and ordering ingredients knowledge of a wide range of recipes familiarity with kitchen sanitation and safety regulations excellent organizational skills

idx: 9898 (Score: 0.4917)
Text:
 iki concepts pte ltd, part-time cook, - roles responsibilities : description ?? minimum 3 days include one saturday or sunday ?? from 1030h to 2130h or 2200h depends on which outlet ? break time from 1500h to 1700h ? monday to friday $12/hr ? saturday to sunday $15/hr ? public holiday $18/hr job summary: the part-time cook shall prep all food ingredients, be able to cook all dishes on the menu, and ensure food prepared is consistent and of superior quality job duties: 1 support the sous chef in ensuring the smooth running of kitchen operations 2 carry out preparation of dishes and ingredients for the shift 3 prepare and execute menu dishes to the highest standards 4 ensure food prepared is consistent and of superior quality 5 carry out daily general kitchen duties which include, but are not limited to, a setting up and cleaning of kitchen equipment and workstations b cleaning of kitchen surfaces, floors, and storage areas c trash disposal 6 organise and ensure kitchen space is clean and tidy in accordance with food safety and hygiene standards 7 assisting other cooks in preparing food or helping other team members when needed 8 any other job-related duties requested from senior staff requirements: - has some kitchen experience - ability to skillfully multitask - attention to detail - works well as part of a team and on individual tasks benefits: ? monthly payout with cpf contributions ? meal provided ?? able to convert to a full-time permanent position

idx: 31238 (Score: 0.4777)
Text:
 iki concepts pte ltd, part-time cook, description ô∏è minimum 3 days include one saturday or sunday ô∏è from 1030h to 2130h or 2200h ‚òï break time from 1500h to 1700h hourly rates: monday to friday $12/hr saturday to sunday $15/hr public holiday $18/hr job summary: the part-time cook shall prep all food ingredients, be able to cook all dishes on the menu, and ensure the food prepared is consistent and of superior quality job duties: 1 support the sous chef in ensuring the smooth running of kitchen operations 2 carry out the preparation of dishes and ingredients for the shift 3 prepare and execute menu dishes to the highest standards 4 ensure food prepared is consistent and of superior quality 5 carry out daily general kitchen duties 6 organise and ensure kitchen space is clean and tidy in accordance with food safety and hygiene standards 7 assisting other cooks in preparing food or helping other team members when needed 8 any other job-related duties requested from senior staff requirements: - has some kitchen experience in restaurants - preferably with relevant experience - ability to skillfully multitask - attention to detail - works well as part of a team and on individual tasks benefits: monthly payout with cpf contributions meal provided ‚úîô∏è able to convert to a full-time permanent position -
```