package com.udacity.webcrawler;

import com.udacity.webcrawler.json.CrawlResult;
import com.udacity.webcrawler.parser.PageParser;
import com.udacity.webcrawler.parser.PageParserFactory;
import javax.inject.Inject;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * A concrete implementation of {@link WebCrawler} that runs multiple threads on a
 * {@link ForkJoinPool} to fetch and process multiple web pages in parallel.
 */
final class ParallelWebCrawler implements WebCrawler {
  private final Clock clock;
  private final Duration timeout;
  private final int popularWordCount;
  private final ForkJoinPool pool;
  private final PageParserFactory pageParserFactory;
  private final List<Pattern> ignoredUrls;
  private final int maxDepth;

  @Inject
  ParallelWebCrawler(Clock clock, @Timeout Duration timeout, @PopularWordCount int popularWordCount,
      @TargetParallelism int threadCount, PageParserFactory pageParserFactory,
      @IgnoredUrls List<Pattern> ignoredUrls, @MaxDepth int maxDepth) {
    this.clock = clock;
    this.timeout = timeout;
    this.popularWordCount = popularWordCount;
    this.pool = new ForkJoinPool(Math.min(threadCount, getMaxParallelism()));
    this.pageParserFactory = pageParserFactory;
    this.ignoredUrls = ignoredUrls;
    this.maxDepth = maxDepth;
  }

  @Override
  public CrawlResult crawl(List<String> startingUrls) {
    // Set the timeout
    Instant deadline = clock.instant().plus(timeout);
    // Use the Concurrent collections to be thread safety
    ConcurrentMap<String, Integer> counts = new ConcurrentSkipListMap<>();
    ConcurrentSkipListSet<String> visitedUrls = new ConcurrentSkipListSet<>();
    // Invoke the Crawl tasks
    startingUrls.forEach(startingUrl -> pool.invoke(new CrawlerTask(clock, timeout, startingUrl,
        deadline, maxDepth, counts, visitedUrls, pageParserFactory, ignoredUrls)));
    // Same in the SequentialWebCrawler, the result shall be sorted out
    if (counts.isEmpty()) {
      return new CrawlResult.Builder().setWordCounts(counts).setUrlsVisited(visitedUrls.size())
          .build();
    }
    return new CrawlResult.Builder().setWordCounts(WordCounts.sort(counts, popularWordCount))
        .setUrlsVisited(visitedUrls.size()).build();
  }

  @Override
  public int getMaxParallelism() {
    return Runtime.getRuntime().availableProcessors();
  }

  private class CrawlerTask extends RecursiveAction {
    private Clock clock;
    private Duration timeout;
    private String startingUrl;
    private Instant deadline;
    private int maxDepth;
    private Map<String, Integer> counts;
    private Set<String> visitedUrls;
    private PageParserFactory pageParserFactory;
    private List<Pattern> ignoredUrls;

    public CrawlerTask(Clock clock, Duration timeout, String startingUrl, Instant deadline,
        int maxDepth, Map<String, Integer> counts, Set<String> visitedUrls,
        PageParserFactory pageParserFactory, List<Pattern> ignoredUrls) {
      this.clock = clock;
      this.timeout = timeout;
      this.startingUrl = startingUrl;
      this.deadline = deadline;
      this.maxDepth = maxDepth;
      this.counts = counts;
      this.visitedUrls = visitedUrls;
      this.pageParserFactory = pageParserFactory;
      this.ignoredUrls = ignoredUrls;
    }

    @Override
    protected void compute() {
      // Check if the maximum depth has been reached or if the deadline has passed
      if (maxDepth == 0 || clock.instant().isAfter(deadline)) {
        return;
      }
      // Check if the starting URL matches any ignored pattern
      if (ignoredUrls.stream().anyMatch(pattern -> pattern.matcher(startingUrl).matches())) {
        return;
      }
      // Check if the starting URL has been visited before
      if (visitedUrls.contains(startingUrl)) {
        return;
      }
      // Mark the starting URL as visited
      if(!visitedUrls.add(startingUrl)) return;

      // Parse the page and retrieve word counts
      PageParser.Result result = pageParserFactory.get(startingUrl).parse();
      // Update word counts
      for (ConcurrentMap.Entry<String, Integer> e : result.getWordCounts().entrySet()) {
        counts.compute(e.getKey(), (k, v) -> (v == null) ? e.getValue() : e.getValue()+v);
    }
      // Create a list to hold subtasks for crawling
      List<CrawlerTask> crawlTasks = new ArrayList<>();
      // Create and add subtasks for crawling each link in the parsed page
      crawlTasks.addAll(result
          .getLinks().stream().map(link -> new CrawlerTask(clock, timeout, link, deadline,
              maxDepth - 1, counts, visitedUrls, pageParserFactory, ignoredUrls))
          .collect(Collectors.toList()));
      // Execute all subtasks in parallel
      invokeAll(crawlTasks);
    }
  }

}


