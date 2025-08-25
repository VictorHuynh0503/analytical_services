import asyncio
import aiohttp
import discord
from discord.ext import tasks
import json
import logging
from datetime import datetime, timezone
import os
from typing import Dict, List, Optional
import feedparser
from dataclasses import dataclass
import nest_asyncio
nest_asyncio.apply()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GoldPrice:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

@dataclass
class NewsEvent:
    title: str
    description: str
    source: str
    url: str
    published: datetime
    category: str = "general"

class EventCrawler:
    def __init__(self):
        self.session = None
        self.news_sources = [
            "http://feeds.bbci.co.uk/news/world/rss.xml",
            "https://feeds.reuters.com/reuters/topNews",
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
        ]
    
    async def init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def fetch_news_events(self) -> List[NewsEvent]:
        """Fetch news events from RSS feeds"""
        events = []
        
        for source_url in self.news_sources:
            try:
                feed = feedparser.parse(source_url)
                source_name = feed.feed.get('title', 'Unknown')
                
                for entry in feed.entries[:5]:  # Limit to 5 latest entries per source
                    try:
                        # Parse publication date
                        pub_date = datetime.now(timezone.utc)
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                        
                        event = NewsEvent(
                            title=entry.get('title', 'No title'),
                            description=entry.get('summary', 'No description')[:200] + "...",
                            source=source_name,
                            url=entry.get('link', ''),
                            published=pub_date,
                            category=self._categorize_news(entry.get('title', ''))
                        )
                        events.append(event)
                    except Exception as e:
                        logger.error(f"Error parsing news entry: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error fetching from {source_url}: {e}")
                continue
        
        return events
    
    def _categorize_news(self, title: str) -> str:
        """Categorize news based on title keywords"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ['economy', 'market', 'trade', 'finance', 'gold', 'stock']):
            return "📈 Finance"
        elif any(word in title_lower for word in ['war', 'conflict', 'military', 'attack']):
            return "⚔️ Conflict"
        elif any(word in title_lower for word in ['election', 'politics', 'government', 'president']):
            return "🏛️ Politics"
        elif any(word in title_lower for word in ['disaster', 'earthquake', 'flood', 'storm']):
            return "🌪️ Disaster"
        else:
            return "🌍 General"

class GoldPriceAPI:
    def __init__(self):
        self.session = None
        # Using a free API for gold prices (you may need to get API key)
        self.api_url = "https://api.metals.live/v1/spot/gold"
        self.backup_url = "https://api.coinbase.com/v2/exchange-rates?currency=XAU"
        
    async def init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def fetch_gold_price(self) -> Optional[GoldPrice]:
        """Fetch current gold price with OHLC data"""
        try:
            # Try primary API
            async with self.session.get(self.api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_metals_live_data(data)
        except Exception as e:
            logger.error(f"Error fetching from primary API: {e}")
        
        try:
            # Try backup API
            async with self.session.get(self.backup_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_coinbase_data(data)
        except Exception as e:
            logger.error(f"Error fetching from backup API: {e}")
        
        return None
    
    def _parse_metals_live_data(self, data: dict) -> GoldPrice:
        """Parse data from metals.live API"""
        timestamp = datetime.now(timezone.utc)
        price = float(data.get('price', 0))
        
        return GoldPrice(
            timestamp=timestamp,
            open=price,  # For real OHLC, you'd need historical data
            high=price,
            low=price,
            close=price
        )
    
    def _parse_coinbase_data(self, data: dict) -> GoldPrice:
        """Parse data from Coinbase API"""
        timestamp = datetime.now(timezone.utc)
        price = float(data['data']['rates']['USD'])
        
        return GoldPrice(
            timestamp=timestamp,
            open=price,
            high=price,
            low=price,
            close=price
        )

class DiscordBot:
    def __init__(self, token: str, channel_id: int):
        self.token = token
        self.channel_id = channel_id
        self.bot = discord.Client(intents=discord.Intents.default())
        self.event_crawler = EventCrawler()
        self.gold_api = GoldPriceAPI()
        self.last_events = set()
        self.last_gold_price = None
        
        # Set up event handlers
        self.bot.event(self.on_ready)
        
    async def on_ready(self):
        """Called when bot is ready"""
        logger.info(f'Bot logged in as {self.bot.user}')
        
        # Initialize sessions
        await self.event_crawler.init_session()
        await self.gold_api.init_session()
        
        # Start scheduled tasks
        self.crawl_and_notify.start()
    
    @tasks.loop(minutes=1)
    async def crawl_and_notify(self):
        """Main loop that crawls data and sends notifications"""
        try:
            channel = self.bot.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return
            
            # Fetch gold price
            gold_price = await self.gold_api.fetch_gold_price()
            if gold_price:
                await self.send_gold_update(channel, gold_price)
            
            # Fetch news events (every 5 minutes to avoid spam)
            if datetime.now().minute % 5 == 0:
                events = await self.event_crawler.fetch_news_events()
                await self.send_news_updates(channel, events)
                
        except Exception as e:
            logger.error(f"Error in crawl_and_notify: {e}")
    
    async def send_gold_update(self, channel, gold_price: GoldPrice):
        """Send gold price update to Discord"""
        try:
            # Calculate price change
            change_emoji = "📊"
            change_text = ""
            
            if self.last_gold_price:
                change = gold_price.close - self.last_gold_price.close
                change_percent = (change / self.last_gold_price.close) * 100
                
                if change > 0:
                    change_emoji = "📈"
                    change_text = f" (+${change:.2f}, +{change_percent:.2f}%)"
                elif change < 0:
                    change_emoji = "📉"
                    change_text = f" (${change:.2f}, {change_percent:.2f}%)"
            
            embed = discord.Embed(
                title=f"{change_emoji} Gold Price Update",
                color=0xFFD700,  # Gold color
                timestamp=gold_price.timestamp
            )
            
            embed.add_field(
                name="Current Price",
                value=f"${gold_price.close:.2f}/oz{change_text}",
                inline=False
            )
            
            embed.add_field(name="Open", value=f"${gold_price.open:.2f}", inline=True)
            embed.add_field(name="High", value=f"${gold_price.high:.2f}", inline=True)
            embed.add_field(name="Low", value=f"${gold_price.low:.2f}", inline=True)
            
            embed.set_footer(text="Updated every minute")
            
            await channel.send(embed=embed)
            self.last_gold_price = gold_price
            
        except Exception as e:
            logger.error(f"Error sending gold update: {e}")
    
    async def send_news_updates(self, channel, events: List[NewsEvent]):
        """Send news updates to Discord"""
        try:
            new_events = []
            
            for event in events:
                event_id = f"{event.title}_{event.source}"
                if event_id not in self.last_events:
                    new_events.append(event)
                    self.last_events.add(event_id)
            
            # Keep only recent events in memory (last 100)
            if len(self.last_events) > 100:
                self.last_events = set(list(self.last_events)[-100:])
            
            for event in new_events[:3]:  # Limit to 3 new events per update
                embed = discord.Embed(
                    title=f"🌍 {event.title}",
                    description=event.description,
                    url=event.url,
                    color=0x3498DB,
                    timestamp=event.published
                )
                
                embed.add_field(
                    name="Category",
                    value=event.category,
                    inline=True
                )
                
                embed.add_field(
                    name="Source",
                    value=event.source,
                    inline=True
                )
                
                embed.set_footer(text="World Events Monitor")
                
                await channel.send(embed=embed)
                
        except Exception as e:
            logger.error(f"Error sending news updates: {e}")
    
    async def start(self):
        """Start the Discord bot"""
        try:
            await self.bot.start(self.token)
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.event_crawler.close_session()
        await self.gold_api.close_session()

# Configuration

from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env into environment

token = os.getenv("DISCORD_TOKEN")
channel = os.getenv("CHANNEL_ID")
DISCORD_TOKEN = token
CHANNEL_ID = channel  # Replace with your channel ID


# DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', 'your_discord_bot_token_here')
# CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', '1234567890'))  # Replace with your channel ID

async def main():
    """Main function to run the bot"""
    bot = DiscordBot(DISCORD_TOKEN, CHANNEL_ID)
    await bot.start()

if __name__ == "__main__":
    # Install required packages:
    # pip install discord.py aiohttp feedparser
    
    print("Starting Discord Crawler Bot...")
    print("Make sure to set environment variables:")
    print("- DISCORD_TOKEN: Your Discord bot token")
    print("- DISCORD_CHANNEL_ID: Channel ID where notifications will be sent")
    
    asyncio.run(main())